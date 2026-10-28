"""NTL classifier fine-tuning with a frozen NF4 backbone and prompt-held-out validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import save_file, load_file
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModel, AutoTokenizer
from train_two_stage import LABELS, LAB2I, Rep


def canonical(text):
    return ' '.join(str(text).split())


def check_data(train, valid):
    for frame in (train, valid):
        assert {'text', 'model', 'prompt_id'}.issubset(frame.columns)
        assert frame[['text', 'model', 'prompt_id']].notna().all().all()
        assert frame.model.isin(LABELS).all()
        assert frame.text.map(canonical).str.len().gt(0).all()
    assert set(valid.model) == set(LABELS), 'validation must cover all 18 labels'
    assert not set(train.text.map(canonical)) & set(valid.text.map(canonical)), 'text leakage'
    assert not set(train.prompt_id.astype(str)) & set(valid.prompt_id.astype(str)), 'prompt leakage'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resume_batches(resume_step, phase_start, epoch, row_count, accum):
    steps = math.ceil(row_count / accum)
    return min(steps, max(0, resume_step - phase_start - epoch * steps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-len', type=int, default=448)
    p.add_argument('--accum', type=int, default=16)
    p.add_argument('--micro-batch', type=int, default=4,
                   help='samples per forward/backward; accum remains samples per optimizer step')
    p.add_argument('--gpu-gib', type=float, default=14)
    p.add_argument('--mixed-epochs', type=int, default=1)
    p.add_argument('--competition-epochs', type=int, default=2)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--external-only-stage1', action='store_true')
    p.add_argument('--class-balanced-loss', action='store_true')
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--allow-optimizer-reset', action='store_true')
    a = p.parse_args()
    assert a.accum > 0 and 0 < a.micro_batch <= a.accum and a.max_len >= 64
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise RuntimeError('Choose a fresh output directory; existing run will not be overwritten')
    resume_meta, resume_step = None, 0
    if a.resume_from:
        assert a.allow_optimizer_reset, 'Weight-only continuation requires explicit optimizer reset acknowledgement'
        resume_meta = json.loads((a.resume_from / 'meta.json').read_text())
        resume_step = int(resume_meta['step'])
        original = json.loads((a.resume_from / 'source_run_manifest.json').read_text())
        for key in ('accum','max_len','mixed_epochs','competition_epochs','external_only_stage1','class_balanced_loss'):
            assert original['config'].get(key) == getattr(a,key), f'Continuation config differs: {key}'
        assert file_hash(a.train) == original['train_sha256'], 'Continuation training data changed'
        assert file_hash(a.valid) == original['valid_sha256'], 'Continuation validation data changed'
        expected_hashes = json.loads((a.resume_from / 'transition_hashes.json').read_text())
        for relative, digest in expected_hashes.items():
            assert file_hash(a.resume_from / relative) == digest, f'Continuation artifact changed: {relative}'
    start = time.time()
    def status(state, **extra):
        value = dict(state=state, elapsed_seconds=round(time.time()-start), pid=os.getpid(),
                     micro_batch=a.micro_batch, effective_batch=a.accum, **extra)
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(out / 'status.json')
    status('checking')
    torch.set_num_threads(4)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    train, valid = pd.read_parquet(a.train), pd.read_parquet(a.valid)
    check_data(train, valid)
    assert 'origin' in train, 'need official competition provenance for adaptation phase'
    comp = np.flatnonzero(train.origin.eq('competition').to_numpy())
    assert len(comp) > 0
    free, total = torch.cuda.mem_get_info()
    budget = min(a.gpu_gib * 1024**3, free - 3 * 1024**3)
    assert budget >= 9 * 1024**3, 'insufficient free GPU memory; other tasks are preserved'
    torch.cuda.set_per_process_memory_fraction(budget / total)
    print(f'GPU free {free/1024**3:.2f} GiB; process allocator cap {budget/1024**3:.2f} GiB', flush=True)
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.base, local_files_only=True, device_map={'': 0},
                                     torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          bias='none', target_modules=['q_proj','k_proj','v_proj','o_proj',
                                                      'gate_proj','up_proj','down_proj']))
    layers = [2,4,6,8,12,16,24,36]
    assert model.config.num_hidden_layers == 36
    rep = Rep(model, layers, model.config.hidden_size)
    rep.norms.to('cuda')
    head = torch.nn.Linear(model.config.hidden_size * len(layers), len(LABELS)).to('cuda').float()
    drop = torch.nn.Dropout(0.1)
    if a.resume_from:
        assert resume_meta['layers'] == layers and resume_meta['hidden'] == model.config.hidden_size
        restored = set_peft_model_state_dict(model, load_file(a.resume_from / 'adapter/adapter_model.safetensors'))
        assert not restored.unexpected_keys, restored.unexpected_keys
        assert not [k for k in restored.missing_keys if '.lora_' in k], 'Adapter weights were not fully restored'
        rep.norms.load_state_dict(load_file(a.resume_from / 'norms.safetensors'), strict=True)
        with np.load(a.resume_from / 'readouts.npz', allow_pickle=False) as saved:
            assert saved['classes'].tolist() == LABELS
            assert saved['plain_coef'].shape == tuple(head.weight.shape)
            with torch.no_grad():
                head.weight.copy_(torch.from_numpy(saved['plain_coef']))
                head.bias.copy_(torch.from_numpy(saved['plain_intercept']))
        print(f'WEIGHT_CONTINUATION step={resume_step} optimizer_reset=true rng_reset=true micro_batch={a.micro_batch}', flush=True)
    adapters = [v for v in model.parameters() if v.requires_grad]
    heads = list(rep.norms.parameters()) + list(head.parameters())
    for v in adapters:
        v.data = v.data.float()
    texts = train.text.tolist() + valid.text.tolist()
    tokens = []
    # Bound tokenizer temporary memory when the external pool contains 150k+ rows.
    for begin in range(0,len(texts),512):
        batch_ids = tok(texts[begin:begin+512],add_special_tokens=False,
                        truncation=True,max_length=a.max_len-1)['input_ids']
        tokens.extend(x+[tok.eos_token_id] for x in batch_ids)
    y = train.model.map(LAB2I).to_numpy()
    yv = valid.model.map(LAB2I).to_numpy()
    def logits(ids, training=False):
        batch = tok.pad({'input_ids': [tokens[int(i)] for i in ids]}, return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = rep(batch)
        return head(drop(z) if training else z)
    def save(name, step, metric=None, phase=None, epoch=None):
        target = out / name
        target.mkdir(exist_ok=not name.startswith(('epoch_','eval_step')))
        model.save_pretrained(target / 'adapter', safe_serialization=True)
        tok.save_pretrained(target / 'tokenizer')
        save_file({k:v.detach().cpu().contiguous() for k,v in rep.norms.state_dict().items()},
                  target / 'norms.safetensors')
        np.savez_compressed(target / 'readouts.npz', classes=np.array(LABELS), pool_layers=np.array(layers),
                plain_coef=head.weight.detach().cpu().numpy(),plain_intercept=head.bias.detach().cpu().numpy())
        (target / 'meta.json').write_text(json.dumps(dict(step=step, balanced_accuracy=metric,
              phase=phase, epoch=epoch,
              base=a.base, layers=layers, keep=36, hidden=model.config.hidden_size,
              max_len=a.max_len, quantization='NF4 double quantization',head='plain',seed=7),indent=2))
    @torch.no_grad()
    def evaluate(step):
        model.eval()
        pred = np.empty((len(valid),len(LABELS)),np.float32)
        for i in range(len(valid)):
            pred[i] = logits([len(train)+i]).float().cpu().numpy()[0]
        score = float(balanced_accuracy_score(yv,pred.argmax(1)))
        np.savez_compressed(out / f'valid_step{step}.npz', logits=pred, y=yv,
                            ids=valid.index.to_numpy(),prompt_id=valid.prompt_id.astype(str).to_numpy())
        print(f'VALID step={step} balanced_accuracy={score:.6f}', flush=True)
        model.train()
        return score
    manifest = dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},train_rows=len(train),valid_rows=len(valid),competition_rows=len(comp),
                    train_sha256=file_hash(a.train),valid_sha256=file_hash(a.valid),labels=LABELS,
                    gpu_allocator_cap_gib=budget/1024**3,torch=torch.__version__,trainer_sha256=file_hash(__file__),
                    continuation_step=resume_step, optimizer_reset=bool(a.resume_from), rng_reset=bool(a.resume_from))
    (out / 'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
    step, best = resume_step, -1.0
    if a.resume_from and (out / 'best/meta.json').exists():
        best = float(json.loads((out / 'best/meta.json').read_text())['balanced_accuracy'])
    if a.resume_from:
        save('start',step,resume_meta.get('balanced_accuracy'))
    rng = np.random.RandomState(7)
    stage1 = np.flatnonzero(train.origin.eq('external').to_numpy()) if a.external_only_stage1 else np.arange(len(train))
    phases = [('external' if a.external_only_stage1 else 'mixed',stage1,a.mixed_epochs,5e-5,5e-4),
              ('competition',comp,a.competition_epochs,2e-5,2e-4)]
    assert 0 <= resume_step < sum(math.ceil(len(rows)/a.accum)*epochs for _,rows,epochs,_,_ in phases)
    phase_start, first_update = 0, True
    try:
        for phase, rows, epochs, lr, hlr in phases:
            if not epochs:
                continue
            counts = np.bincount(y[rows],minlength=len(LABELS))
            assert (counts>0).all(), f'{phase}: missing class'
            weights = len(rows)/(len(LABELS)*counts) if a.class_balanced_loss else np.ones(len(LABELS))
            opt = torch.optim.AdamW([{'params':adapters,'lr':lr,'initial_lr':lr},
                                    {'params':heads,'lr':hlr,'initial_lr':hlr}],weight_decay=.01)
            total_steps = math.ceil(len(rows)/a.accum)*epochs
            warm = max(1,int(total_steps*.05))
            def factor(s):
                return min(1.,(s+1)/warm) if s<warm else .1+.9*.5*(1+math.cos(math.pi*(s-warm)/max(1,total_steps-warm)))
            completed = min(total_steps,max(0,resume_step-phase_start))
            sched = torch.optim.lr_scheduler.LambdaLR(opt,factor,last_epoch=completed-1)
            model.train()
            for epoch in range(epochs):
                plan = rng.permutation(rows)
                skipped = resume_batches(resume_step,phase_start,epoch,len(rows),a.accum)
                if skipped == math.ceil(len(rows)/a.accum):
                    continue
                assert step == phase_start + epoch*math.ceil(len(rows)/a.accum) + skipped
                opt.zero_grad(set_to_none=True)
                for begin in range(skipped*a.accum,len(plan),a.accum):
                    block = plan[begin:begin+a.accum]
                    loss_value = 0.
                    tick = time.time()
                    for micro_begin in range(0,len(block),a.micro_batch):
                        micro = block[micro_begin:micro_begin+a.micro_batch]
                        scores = logits(micro,training=True)
                        losses = torch.nn.functional.cross_entropy(scores,
                            torch.tensor(y[micro],device='cuda'),reduction='none')
                        sample_weights = torch.tensor(weights[y[micro]],device='cuda',dtype=losses.dtype)
                        loss = (losses * sample_weights).sum()/len(block)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'nonfinite loss at rows {micro}')
                        loss.backward()
                        loss_value += loss.item()
                    norm = torch.nn.utils.clip_grad_norm_(adapters+heads,1.0,error_if_nonfinite=True)
                    if first_update:
                        assert any(v.grad is not None and v.grad.abs().sum().item()>0 for v in adapters), 'LoRA received no gradients'
                        first_update = False
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    step += 1
                    gpu = torch.cuda.max_memory_reserved()/1024**3
                    print(f'TRAIN {phase} epoch={epoch+1} step={step} rows={begin+len(block)}/{len(plan)} '
                          f'loss={loss_value:.4f} grad={norm:.3f} seconds={time.time()-tick:.1f} gpu_peak={gpu:.2f}GiB '
                          f'micro_batch={a.micro_batch} lr={opt.param_groups[0]["lr"]:.8g}',flush=True)
                    status('training',phase=phase,epoch=epoch+1,step=step,loss=loss_value,gpu_peak_gib=gpu)
                    if a.smoke_steps and step>=a.smoke_steps:
                        metric = evaluate(step)
                        save('smoke',step,metric)
                        status('smoke_complete',step=step,gpu_peak_gib=gpu)
                        return
                    if step%100 == 0:
                        save('latest',step)
                    if a.eval_every and step%a.eval_every == 0:
                        metric = evaluate(step)
                        save(f'eval_step{step}',step,metric,phase,epoch+1)
                        if metric>best:
                            best=metric
                            save('best',step,metric)
                metric = evaluate(step)
                save(f'epoch_{phase}_{epoch+1:02d}_step{step}',step,metric,phase,epoch+1)
                save('latest',step,metric)
                if metric>best:
                    best=metric
                    save('best',step,metric)
                status('training',phase=phase,step=step,best_balanced_accuracy=best)
            phase_start += total_steps
        status('complete',step=step,best_balanced_accuracy=best)
    except Exception as error:
        status('failed',step=step,error=repr(error))
        raise


if __name__ == '__main__':
    main()
