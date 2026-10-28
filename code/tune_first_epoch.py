"""Sequential, recorded first-epoch trials; only a passing trial continues to epoch20."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'runs/tune_first_epoch_20261027'
TARGET=0.5777777777777777
TRIALS=[
    dict(name='01_original_schedule',seed=7,advance=153261,lr=2e-5,head=2e-4,warm=13,beta1=.9,beta2=.999,eps=1e-8,wd=.01),
    dict(name='02_fresh_shuffle',seed=7,advance=0,lr=2e-5,head=2e-4,warm=13,beta1=.9,beta2=.999,eps=1e-8,wd=.01),
    dict(name='03_seed11',seed=11,advance=153261,lr=2e-5,head=2e-4,warm=13,beta1=.9,beta2=.999,eps=1e-8,wd=.01),
    dict(name='04_half_lr',seed=7,advance=153261,lr=1e-5,head=1e-4,warm=13,beta1=.9,beta2=.999,eps=1e-8,wd=.01),
    dict(name='05_adam_short_warm',seed=7,advance=153261,lr=2e-5,head=1e-4,warm=5,beta1=.9,beta2=.99,eps=1e-6,wd=.001),
    dict(name='06_conservative',seed=42,advance=153261,lr=1e-5,head=1e-4,warm=5,beta1=.95,beta2=.999,eps=1e-8,wd=0.),
]

def main():
    WORK.mkdir(exist_ok=True)
    lock=(WORK/'tuning.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_file=WORK/'tuning_status.json'
    assert not state_file.exists(),'Existing tuning session; inspect rather than duplicating trials'
    state=dict(target=TARGET,trials=[],selection_bias_warning='Repeated use of the same 540-row holdout; not an independent evaluation')
    def save(**values):
        state.update(values)
        temp=state_file.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2));temp.replace(state_file)
        print(json.dumps(values),flush=True)
    source=ROOT/'runs/large_mb4_20261027/epoch_external_01_step9579'
    assert json.loads((source/'meta.json').read_text())['step']==9579
    assert json.loads((ROOT/'runs/official20_from9579_20261027/status.json').read_text())['state']=='stopped'
    env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for trial in TRIALS:
        assert shutil.disk_usage(ROOT).free>=4.3*1024**3,'Need room for passing trial and its 20 checkpoints'
        out=WORK/trial['name']
        assert not out.exists(),out
        cmd=[sys.executable,'-u',str(ROOT/'train_qlora.py'),
             '--base',str(ROOT/'models/qwen3-8b-base-nf4'),
             '--train',str(ROOT/'data/official20_20261027/train_official.parquet'),
             '--valid',str(ROOT/'data/val_curated.parquet'),'--out',str(out),'--init-from',str(source),
             '--mixed-epochs','0','--competition-epochs','20','--micro-batch','4','--accum','16','--gpu-gib','14',
             '--class-balanced-loss','--schedule-steps','270','--first-epoch-min-bacc',str(TARGET),
             '--seed',str(trial['seed']),'--shuffle-advance',str(trial['advance']),
             '--competition-adapter-lr',str(trial['lr']),'--competition-head-lr',str(trial['head']),
             '--warmup-steps',str(trial['warm']),'--adam-beta1',str(trial['beta1']),
             '--adam-beta2',str(trial['beta2']),'--adam-eps',str(trial['eps']),'--weight-decay',str(trial['wd'])]
        log_path=WORK/(trial['name']+'.log')
        with log_path.open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(state='trial_running',active_trial=trial,active_run=str(out),active_pid=child.pid,log=str(log_path))
            selected=False
            deadline=time.monotonic()+24*3600
            while child.poll() is None:
                gate_file=out/'first_epoch_gate.json'
                if gate_file.exists() and not selected:
                    try:
                        gate=json.loads(gate_file.read_text())
                    except json.JSONDecodeError:
                        time.sleep(2);continue
                    if gate['passed']:
                        selected=True
                        save(state='selected_training',selected_trial=trial,selected_run=str(out),first_epoch=gate)
                if time.monotonic()>deadline:
                    child.terminate();child.wait(timeout=60)
                    save(state='failed',reason='24h trial limit exceeded');return 1
                time.sleep(10)
            result=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
            gate_file=out/'first_epoch_gate.json'
            gate=json.loads(gate_file.read_text()) if gate_file.exists() else None
            state['trials'].append(dict(config=trial,exit_code=child.returncode,status=result,first_epoch=gate))
            save(last_trial=trial['name'])
            if child.returncode!=0:
                save(state='failed',reason='Trial process failed; inspect log before retrying');return 1
            if gate and gate['passed']:
                save(state='complete',selected_trial=trial,selected_run=str(out),first_epoch=gate)
                return 0
            assert result.get('state')=='first_epoch_rejected',result
    save(state='target_not_met',reason='All six predefined trials below target; no result fabricated')
    return 2

if __name__=='__main__':
    raise SystemExit(main())
