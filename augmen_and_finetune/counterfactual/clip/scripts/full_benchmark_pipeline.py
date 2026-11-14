#!/usr/bin/env python3
"""Persistent full benchmark execution with stage logs and explicit completion."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/benchmark_full_20261027'
DATA=ROOT/'data/benchmark_full_20261027'
CACHE=ROOT/'data/benchmark_parquet_20261027'
PY=sys.executable
STATE=OUT/'PIPELINE_STATUS.json'
REV='182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1'


def state(stage,status,**extra):
    value={'stage':stage,'status':status,'pid':os.getpid(),'updated_at_unix':time.time(),**extra}
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)
    print(json.dumps(value),flush=True)


def run(stage,command,complete,skip_existing=True):
    if skip_existing and complete.exists():return
    if stage in {'baseline', 'adapted'} and not complete.exists():
        partial = OUT/f'{stage}.csv'
        if partial.exists(): partial.rename(OUT/f'{stage}.interrupted-{int(time.time())}.csv')
    state(stage,'running',command=command)
    with (OUT/f'{stage}.log').open('a') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed with exit {result.returncode}; see its log')
    if not complete.exists():raise RuntimeError(f'{stage} exited without expected completion artifact: {complete}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--download-pid',type=int);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    caffeinate=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]) if shutil.which('caffeinate') else None
    try:
        if args.download_pid and not (CACHE/'DOWNLOAD_COMPLETE.json').exists():
            state('download','running',download_pid=args.download_pid)
            while not (CACHE/'DOWNLOAD_COMPLETE.json').exists():
                try:os.kill(args.download_pid,0)
                except ProcessLookupError:break
                time.sleep(15)
        for attempt in range(3):
            try:
                run('download',[PY,'-u','scripts/fetch_benchmark_ranges.py'],CACHE/'DOWNLOAD_COMPLETE.json')
                break
            except RuntimeError:
                if attempt == 2: raise
                time.sleep(15)
        run('preprocess',[PY,'-u','scripts/team_data_local.py','preprocess_dataset','--output',str(DATA),'--revision',REV],DATA/'preprocessing_report.json')
        run('splits',[PY,'-u','benchmark_source/data/create_splits.py','--input',str(DATA/'metadata.csv'),'--output',str(DATA/'splits'),'--seed','42'],DATA/'splits/split_report.json')
        run('export',[PY,'-u','scripts/export_benchmark_parallel.py'],DATA/'exported/export_report.json')
        run('import',[PY,'-u','import_benchmark.py','--exported',str(DATA/'exported'),'--output',str(OUT/'processed')],OUT/'processed/summary.json')
        cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
        cfg['data']={'source':'team_exported_csv','processed_dir':str(OUT/'processed'),'image_dir':str(DATA/'exported/images')}
        cfg['experiment']['status']='FULL_TEAM_BENCHMARK_SINGLE_SEED_CLIP_LORA'
        cfg['training'].update(output_dir=str(OUT/'training'),resume=True,checkpoint_steps=250)
        cfg['prediction']['batch_size']=64
        (OUT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        run('baseline',[PY,'-u','predict.py','--config',str(OUT/'config.yaml'),'--output',str(OUT/'baseline.csv')],OUT/'baseline.metadata.json')
        # Retry interruption only from an actual optimizer checkpoint.
        if not (OUT/'training/TRAINING_COMPLETE.json').exists():
            old_metadata = OUT/'training/metadata.json'
            already_finished = old_metadata.exists() and json.loads(old_metadata.read_text()).get('adapter_reload_check',{}).get('passed')
            if not already_finished:
                run('training',[PY,'-u','train_lora.py','--config',str(OUT/'config.yaml')],old_metadata,skip_existing=False)
            meta=json.loads((OUT/'training/metadata.json').read_text())
            if not meta.get('adapter_reload_check',{}).get('passed'):raise RuntimeError('Training reload validation is incomplete')
            (OUT/'training/TRAINING_COMPLETE.json').write_text(json.dumps({'passed':True,'optimizer_steps':meta['optimizer_steps']}))
        run('adapted',[PY,'-u','predict.py','--config',str(OUT/'config.yaml'),'--adapter',str(OUT/'training/best_adapter'),'--output',str(OUT/'adapted.csv')],OUT/'adapted.metadata.json')
        run('evaluation',[PY,'-u','evaluate_predictions.py','--baseline',str(OUT/'baseline.csv'),'--adapted',str(OUT/'adapted.csv'),'--output',str(OUT/'comparison.json')],OUT/'comparison.json')
        run('capability',[PY,'-u','evaluate_capability.py','--config',str(OUT/'config.yaml'),'--adapter',str(OUT/'training/best_adapter'),'--output-dir',str(OUT/'capability')],OUT/'capability/comparison.json')
        from datetime import datetime,timezone
        result=json.loads((OUT/'comparison.json').read_text())
        cap=json.loads((OUT/'capability/comparison.json').read_text())
        prep=json.loads((DATA/'preprocessing_report.json').read_text())
        report=f'''# Full benchmark results\n\nCompleted UTC: {datetime.now(timezone.utc).isoformat()}\n\nDataset: Intel/SocialCounterfactuals at {REV}. Total {prep['total_rows']} images, {prep['counterfactual_groups']} matched groups, {prep['unique_occupations']} occupations. Team split algorithm, seed 42; all source groups retained. CLIP ViT-B/32, visual q/v LoRA, 3 epochs, CE + 0.1 JS, validation loss selection.\n\n| Metric | Baseline | LoRA |\n|---|---:|---:|\n'''
        for key in ('accuracy','group_balanced_accuracy','all_variants_agreement','group_mean_pairwise_agreement'):
            report+=f"| {key} | {result['baseline'][key]:.6f} | {result['adapted'][key]:.6f} |\n"
        report+=f"\nCIFAR-10 external image-classification retention proxy: {cap['baseline']['accuracy']:.4%} → {cap['adapted']['accuracy']:.4%}. Source: https://cave.cs.toronto.edu/kriz/cifar.html (Krizhevsky, 2009). This is one proxy, not comprehensive general capability.\n\nSubgroup counts and gaps are in comparison.json; differences are descriptive, not causal fairness results. Single training seed; no other models/methods or multi-seed comparison claimed. Test data were not used for checkpoint selection.\n"
        (OUT/'RESULTS.md').write_text(report)
        state('complete','complete',report=str(OUT/'RESULTS.md'))
    except BaseException as exc:
        state('failed','failed',error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise
    finally:
        if caffeinate is not None:
            caffeinate.terminate()

if __name__=='__main__':main()
