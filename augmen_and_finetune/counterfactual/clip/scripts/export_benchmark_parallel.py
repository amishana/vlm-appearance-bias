#!/usr/bin/env python3
"""Parallel equivalent of the team's PNG exporter against pinned local shards.

Uses the team's CSV schema, sequential source row indices, field checks,
relative PNG paths and fixed splits. Images are unchanged RGB conversions.
"""
import csv
import hashlib
import io
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/benchmark_parquet_20261027'
DATA = ROOT/'data/benchmark_full_20261027'
OUT = DATA/'exported'


def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_csv(p,rows,columns):
    tmp=p.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    tmp.replace(p)


def main():
    manifest=json.loads((CACHE/'source_manifest.json').read_text())
    if not (CACHE/'DOWNLOAD_COMPLETE.json').exists():raise RuntimeError('Download incomplete')
    metadata=read_csv(DATA/'metadata.csv'); columns=list(metadata[0]); hashes={}; OUT.mkdir(exist_ok=True)
    images=OUT/'images';images.mkdir(exist_ok=True)
    def export(item):
        idx,source=item;row=metadata[idx]
        if idx!=int(row['source_row_index']):raise ValueError('Nonsequential source indices')
        for key in ('caption','a1','a2','counterfactual_set','dataset_segment','a1_type','a2_type'):
            if str(source[key])!=row[key]:raise ValueError(f'Source metadata mismatch at {idx}: {key}')
        path=images/f"{row['image_id']}.png"
        if not path.exists():
            with Image.open(io.BytesIO(source['image']['bytes'])) as image:
                image=image.convert('RGB'); temporary=path.with_suffix('.tmp');image.save(temporary,format='PNG');temporary.replace(path)
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        row['image_path']='images/'+path.name
        return row['image_id'],sha,path.stat().st_size
    idx=0; total_bytes=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number,shard in enumerate(manifest['shards'],1):
            if shutil.disk_usage(OUT).free < 5*1024**3:
                raise RuntimeError('Less than 5 GiB free; export paused before disk exhaustion')
            file=pq.ParquetFile(CACHE/shard['path'])
            for batch in file.iter_batches(batch_size=128):
                values=batch.to_pylist()
                result=pool.map(export,enumerate(values,idx))
                for sample,sha,size in result:hashes[sample]=sha;total_bytes+=size
                idx+=len(values)
                if idx%1024==0: print(json.dumps({'exported_images':idx,'total':len(metadata),'bytes':total_bytes}),flush=True)
            progress={'shards':number,'total_shards':len(manifest['shards']),'images':idx,'total_images':len(metadata),'png_bytes':total_bytes}
            (OUT/'progress.json').write_text(json.dumps(progress))
            print(json.dumps(progress),flush=True)
    if idx!=len(metadata):raise ValueError('Source and metadata lengths differ')
    write_csv(OUT/'metadata_with_images.csv',metadata,columns)
    by_id={r['image_id']:r for r in metadata}
    for name in ('train','val','test'):
        rows=read_csv(DATA/'splits'/f'{name}.csv')
        for row in rows:row['image_path']=by_id[row['image_id']]['image_path']
        write_csv(OUT/f'{name}.csv',rows,list(rows[0]))
    with (OUT/'image_hashes.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','sha256']);w.writerows(sorted(hashes.items()))
    (OUT/'export_report.json').write_text(json.dumps({'dataset':manifest['dataset'],'revision':manifest['revision'],
       'selected_images':idx,'complete':True,'png_bytes':total_bytes,'implementation':'parallel PNG equivalent; original source field validation; cached pinned shards'},indent=2))

if __name__=='__main__':main()
