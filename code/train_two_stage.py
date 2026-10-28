"""Two-stage LoRA classifier for Name That LLM  (server edition, phased).

Phase s1 : train on the public pool only (learns what each model sounds like) -> saves adapter
Phase s2 : load the s1 adapter, fine-tune on the host train rows  -> saves adapter + heads

Validation: --job 0..4 holds that prompt fold out of stage 2 (stage 1 already never sees the host
data), and reports balanced accuracy on it.  --job final trains on everything.

Recipe (LoRA24 lineage + two-stage fine-tuning):
  * Qwen3-8B-Base truncated to its first `keep` layers
  * LoRA r16 / alpha32 / dropout .05 on q,k,v,o,gate,up,down
  * masked mean pool (first token dropped) over 8 layers, one LayerNorm each, concatenated
  * two linear heads: plain and centred (centred = trained on same-prompt triplet-centred features)
  * triplets of same-prompt passages, `accum` triplets per optimizer step
  * stage 1: class-weighted CE over the pool, lr 1e-4 / 1e-3
  * stage 2: CE over the host rows, lr 5e-5 / 5e-4
  * bf16 autocast, fp32 adapters/heads, gradient checkpointing, max_len 447 + EOS
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
LAB2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--pool', default='')
    p.add_argument('--out', required=True)
    p.add_argument('--phase', required=True, choices=['s1', 's2'])
    p.add_argument('--job', default='final', help='fold number 0-4 or "final" (stage 2 only)')
    p.add_argument('--init-adapter', default='', help='adapter dir to load (s1 -> s2)')
    p.add_argument('--layers', default='2,4,6,8,12,16,20,24')
    p.add_argument('--keep', type=int, default=24)
    p.add_argument('--max-len', type=int, default=447)
    p.add_argument('--epochs', type=int, default=0, help='default: s1=1, s2=2')
    p.add_argument('--accum', type=int, default=6)
    p.add_argument('--lr-adapter', type=float, default=-1)
    p.add_argument('--lr-head', type=float, default=-1)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--r', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--ldrop', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--fold-seed', type=int, default=0)
    p.add_argument('--pool-cap', type=int, default=0)
    p.add_argument('--no-ckpt', action='store_true', help='disable gradient checkpointing (32GB GPUs)')
    p.add_argument('--stage1-with-train', action='store_true')
    p.add_argument('--class-weight-s1', type=int, default=1)
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    return p.parse_args()


class Rep(nn.Module):
    def __init__(self, model, layers, hidden):
        super().__init__()
        self.model = model
        self.layers = layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in layers])

    def forward(self, batch):
        hs = self.model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        m = batch['attention_mask'].unsqueeze(-1).float()
        m[:, 0] = 0
        with torch.autocast('cuda', enabled=False):
            z = torch.cat([nm((hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)) for l, nm in zip(self.layers, self.norms)], -1)
        return z


def triplets(rows, pid_all, rng, accum):
    groups = {}
    for r in rows:
        groups.setdefault(pid_all[r], []).append(int(r))
    out, loose = [], []
    for p, members in groups.items():
        m = rng.permutation(members)
        if len(m) < 3:
            loose.extend(int(v) for v in m)
            continue
        for s in range(0, len(m), 3):
            b = list(m[s:s + 3])
            while len(b) < 3:
                b.append(int(rng.choice([x for x in m if x not in b])))
            out.append(b)
    loose = list(rng.permutation(loose))
    out += [loose[s:s + 3] for s in range(0, len(loose) - 2, 3)]
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % accum:
        out.append(out[rng.randint(len(out))])
    return out


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    dev = torch.device('cuda:0')
    layers = [int(x) for x in a.layers.split(',')]
    t0 = time.time()
    if a.epochs <= 0:
        a.epochs = 1 if a.phase == 's1' else 2
    if a.lr_adapter < 0:
        a.lr_adapter = 1e-4 if a.phase == 's1' else 5e-5
    if a.lr_head < 0:
        a.lr_head = 1e-3 if a.phase == 's1' else 5e-4

    train = pd.read_csv(a.train)
    y = train['model'].map(LAB2I).values
    pid = train['prompt_id'].values
    NTR = len(train)

    if a.phase == 's1':
        pool = pd.read_parquet(a.pool)
        pool = pool[pool['model'].isin(LAB2I)].reset_index(drop=True)
        if a.pool_cap:
            parts = [g if len(g) <= a.pool_cap else g.sample(a.pool_cap, random_state=0) for _, g in pool.groupby('model')]
            pool = pd.concat(parts).reset_index(drop=True)
        yp = pool['model'].map(LAB2I).values
        pidp = pool['prompt_id'].astype(str).values
        rows = np.arange(NTR, NTR + len(pool))
        if a.stage1_with_train:
            rows = np.concatenate([np.arange(NTR), rows])
        pool_cnt = np.bincount(yp, minlength=K).astype(float)
    else:
        pool, yp, pidp, pool_cnt = None, None, None, None
        rows = None

    if a.job == 'final' or a.phase == 's1':
        fit = np.arange(NTR)
        valid = np.array([], dtype=int)
    else:
        j = int(a.job)
        prompts = np.array(sorted(train['prompt_id'].unique()))
        rng0 = np.random.RandomState(a.fold_seed)
        rng0.shuffle(prompts)
        fold = train['prompt_id'].map({p: i % 5 for i, p in enumerate(prompts)}).values
        fit = np.where(fold != j)[0]
        valid = np.where(fold == j)[0]
    if a.phase == 's2':
        rows = fit

    if a.phase == 's1':
        y_all = np.concatenate([y, yp])
        pid_all = np.concatenate([pid, pidp])
        texts = train['text'].astype(str).tolist() + pool['text'].astype(str).tolist()
    else:
        y_all, pid_all = y, pid
        texts = train['text'].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mc = AutoConfig.from_pretrained(a.base)
    mc.num_hidden_layers, mc.use_cache = a.keep, False
    if getattr(mc, 'layer_types', None) is not None:
        mc.layer_types = mc.layer_types[:a.keep]
    dtype = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    base = AutoModel.from_pretrained(a.base, config=mc, dtype=dtype, attn_implementation='sdpa')
    base.to(dev)
    if not a.no_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
    if a.init_adapter:
        model = PeftModel.from_pretrained(base, a.init_adapter, is_trainable=True)
        # re-apply fp32 for trainable params
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        model = get_peft_model(base, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.ldrop, bias='none',
                                               target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    hidden = mc.hidden_size
    rep = Rep(model, layers, hidden).to(dev)
    if a.init_adapter and os.path.exists(os.path.join(a.init_adapter, 'norms.safetensors')):
        rep.norms.load_state_dict(load_file(os.path.join(a.init_adapter, 'norms.safetensors')))
        print(f'  warm-started feature norms from {a.init_adapter}', flush=True)
    plain = nn.Linear(hidden * len(layers), K).to(dev).float()
    centred = nn.Linear(hidden * len(layers), K).to(dev).float()
    drop = nn.Dropout(0.1)
    ad = [p for p in model.parameters() if p.requires_grad]
    hd = list(rep.norms.parameters()) + list(plain.parameters()) + list(centred.parameters())
    opt = torch.optim.AdamW([{'params': ad, 'lr': a.lr_adapter}, {'params': hd, 'lr': a.lr_head}], weight_decay=a.wd)
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=(a.dtype == 'fp16'))
    amp = torch.autocast('cuda', dtype=dtype)

    cls_w = None
    if a.phase == 's1' and a.class_weight_s1:
        w = pool_cnt.sum() / (K * np.maximum(pool_cnt, 1))
        cls_w = torch.tensor(w, dtype=torch.float32, device=dev)

    print(f'[{a.phase}/{a.job}] train {NTR} pool {0 if pool is None else len(pool)} | fit {len(fit)} valid {len(valid)} '
          f'| epochs {a.epochs} lr {a.lr_adapter}/{a.lr_head} | {time.time() - t0:.0f}s', flush=True)

    tokens = [ids + [tok.eos_token_id] for ids in
              tok(texts, add_special_tokens=False, truncation=True, max_length=a.max_len)['input_ids']]
    print(f'  tokenized {len(tokens)} in {time.time() - t0:.0f}s', flush=True)

    def z_of(idx):
        batch = tok.pad({'input_ids': [tokens[i] for i in idx]}, return_tensors='pt').to(dev)
        with amp:
            z = rep(batch)
        return z

    @torch.no_grad()
    def evaluate(tag):
        model.eval()
        if not len(valid):
            return None, float('nan')
        Z = np.zeros((len(valid), hidden * len(layers)), np.float32)
        order = np.argsort([len(tokens[valid[i]]) for i in range(len(valid))], kind='stable')
        for s in range(0, len(order), 6):
            pos = order[s:s + 6]
            Z[pos] = z_of(valid[pos]).float().cpu().numpy()
        Wc = plain.weight.detach().cpu().numpy()
        bc = plain.bias.detach().cpu().numpy()
        pred = (Z @ Wc.T + bc).argmax(1)
        acc = float((pred == y[valid]).mean())
        from sklearn.metrics import balanced_accuracy_score
        bacc = float(balanced_accuracy_score(y[valid], pred))
        print(f'  [{tag}] valid acc {acc:.4f} balanced {bacc:.4f} ({time.time() - t0:.0f}s)', flush=True)
        per = {LABELS[c]: round(float((pred[y[valid] == c] == c).mean()), 3) for c in range(K) if (y[valid] == c).sum()}
        print('  per-class: ' + json.dumps(per), flush=True)
        return Z, bacc

    rng = np.random.RandomState(a.seed + (11 if a.phase == 's1' else 22))
    for ep in range(a.epochs):
        plan = triplets(rows, pid_all, rng, a.accum)
        # length-bucket the plan: each accumulation group gets triplets of similar token length,
        # which removes most of the padding waste
        lens = np.array([max(len(tokens[i]) for i in b) for b in plan])
        plan = [plan[i] for i in np.argsort(lens, kind='stable')]
        total = len(plan) // a.accum
        warm = max(1, int(a.warmup * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        print(f'[{a.phase}] epoch {ep}: {len(plan)} triplets, {total} steps', flush=True)
        model.train()
        losses = []
        step = 0
        for i, b in enumerate(plan):
            z = z_of(b)
            yy = torch.as_tensor(y_all[b], device=dev)
            if z.shape[0] % 3 == 0:
                # centre inside each same-prompt triplet, matching the test-time group centring
                zt = z.view(-1, 3, z.shape[-1])
                zc = (zt - zt.mean(1, keepdim=True)).reshape(z.shape[0], -1)
            else:
                zc = z - z.mean(0, keepdim=True)
            loss = F.cross_entropy(plain(drop(z)), yy, weight=cls_w)
            loss = loss + F.cross_entropy(centred(drop(zc)), yy, weight=cls_w)
            scaler.scale(loss / a.accum).backward()
            losses.append(float(loss.detach()))
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ad + hd, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 50 == 0:
                    print(f'  ep{ep} step {step}/{total} loss {np.mean(losses[-200:]):.4f} ({time.time() - t0:.0f}s)', flush=True)
        if step % 50:
            print(f'  ep{ep} step {step}/{total} loss {np.mean(losses[-200:]):.4f} ({time.time() - t0:.0f}s)', flush=True)
        if len(valid):
            evaluate(f'ep{ep}')

    # ---- outputs
    os.makedirs(a.out, exist_ok=True)
    if a.phase == 's2' and len(valid):
        Zv, bacc = evaluate('final')
        np.savez_compressed(os.path.join(a.out, 'valid_features.npz'),
                            ids=train['id'].values[valid], features=Zv.astype(np.float16), y=y[valid])
    else:
        bacc = float('nan')
    W = {n: t.weight.detach().cpu().numpy() for n, t in (('plain', plain), ('center', centred))}
    B = {n: t.bias.detach().cpu().numpy() for n, t in (('plain', plain), ('center', centred))}
    np.savez_compressed(os.path.join(a.out, 'readouts.npz'), classes=np.array(LABELS), pool_layers=np.array(layers),
                        plain_coef=W['plain'], plain_intercept=B['plain'], center_coef=W['center'], center_intercept=B['center'])
    save_file({k: v.detach().cpu().contiguous() for k, v in rep.norms.state_dict().items()},
              os.path.join(a.out, 'norms.safetensors'))
    model.save_pretrained(os.path.join(a.out, 'adapter'), safe_serialization=True)
    tok.save_pretrained(os.path.join(a.out, 'tokenizer'))
    meta = dict(phase=a.phase, job=a.job, labels=LABELS, layers=layers, keep=a.keep, max_len=a.max_len, hidden=hidden,
                cfg=vars(a), balanced_acc=bacc, secs=time.time() - t0,
                pool_rows=int(0 if pool is None else len(pool)), train_rows=int(NTR), valid_rows=int(len(valid)))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(f'[{a.phase}/{a.job}] DONE balanced {bacc:.4f} in {time.time() - t0:.0f}s -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
