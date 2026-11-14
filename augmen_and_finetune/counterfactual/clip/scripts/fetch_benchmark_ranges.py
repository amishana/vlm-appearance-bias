#!/usr/bin/env python3
"""Resumable HTTP range download of immutable official shards, with LFS SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import requests
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/benchmark_parquet_20261027'
REV='182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1'
REPO='Intel/SocialCounterfactuals'
CHUNK=16*1024*1024
LOCAL=threading.local()


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    manifest_path=DEST/'source_manifest.json'
    if manifest_path.exists():manifest=json.loads(manifest_path.read_text())
    else:
        info=HfApi().dataset_info(REPO,revision=REV,files_metadata=True)
        manifest={'dataset':REPO,'revision':info.sha,'shards':[{'path':s.rfilename,'bytes':s.size,'sha256':s.lfs.sha256}
          for s in sorted(info.siblings,key=lambda x:x.rfilename) if s.rfilename.endswith('.parquet')]}
        atomic_json(manifest_path,manifest)
    if manifest['revision']!=REV:raise ValueError('Manifest revision differs')
    futures={};complete=set();downloaded=0;lock=threading.Lock();started=time.monotonic()
    def fetch(shard,start,end):
        nonlocal downloaded
        if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
        chunk_dir=DEST/'.chunks'/Path(shard['path']).stem;chunk_dir.mkdir(parents=True,exist_ok=True)
        path=chunk_dir/f'{start:012d}.bin';length=end-start+1
        if not path.exists() or path.stat().st_size!=length:
            for attempt in range(8):
                try:
                    tmp=path.with_suffix('.tmp')
                    received=tmp.stat().st_size if tmp.exists() else 0
                    if received > length:
                        tmp.unlink(); received=0
                    if received == length:
                        tmp.replace(path); break
                    next_start=start+received
                    url=f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{shard['path']}?download=true&range_start={next_start}"
                    with LOCAL.session.get(url,headers={'Range':f'bytes={next_start}-{end}'},timeout=(20,90),stream=True) as r:
                        r.raise_for_status()
                        if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {next_start}-{end}/'):
                            raise ValueError('Server did not honor requested byte range')
                        with tmp.open('ab' if received else 'wb') as f:
                            for data in r.iter_content(1024*1024):f.write(data)
                        if tmp.stat().st_size!=length:raise ValueError('Truncated byte range')
                        tmp.replace(path)
                    break
                except Exception as exc:
                    # URLs can carry signed query parameters; log only exception type.
                    print(json.dumps({'retry_shard':Path(shard['path']).name,'offset':start,'attempt':attempt+1,'error_type':type(exc).__name__}),flush=True)
                    if attempt==7:raise RuntimeError(f"Range download failed for {shard['path']} at {start}") from None
                    time.sleep(min(2**attempt,30))
        with lock:downloaded+=length
        return path
    with ThreadPoolExecutor(max_workers=24) as pool:
        for shard in manifest['shards']:
            target=DEST/shard['path']
            if target.is_file() and target.stat().st_size==shard['bytes'] and sha(target)==shard['sha256']:
                complete.add(shard['path']);continue
            for start in range(0,shard['bytes'],CHUNK):
                end=min(start+CHUNK,shard['bytes'])-1
                future=pool.submit(fetch,shard,start,end);futures[future]=(shard,start)
        remaining={s['path']:sum(1 for _ in range(0,s['bytes'],CHUNK)) for s in manifest['shards'] if s['path'] not in complete}
        for future in as_completed(futures):
            chunk=future.result();shard,_=futures[future];remaining[shard['path']]-=1
            if remaining[shard['path']]==0:
                target=DEST/shard['path'];target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix('.tmp');chunk_dir=chunk.parent
                with tmp.open('wb') as output:
                    for part in sorted(chunk_dir.glob('*.bin')):
                        with part.open('rb') as f:
                            while data:=f.read(4*1024*1024):output.write(data)
                if tmp.stat().st_size!=shard['bytes'] or sha(tmp)!=shard['sha256']:raise ValueError('Assembled shard failed SHA256')
                tmp.replace(target)
                for part in chunk_dir.glob('*.bin'):part.unlink()
                complete.add(shard['path'])
                print(json.dumps({'verified_shard':Path(shard['path']).name,'complete':len(complete),'total':len(manifest['shards'])}),flush=True)
            atomic_json(DEST/'download_progress.json',{'verified_shards':len(complete),'total_shards':len(manifest['shards']),
                 'fetched_bytes_this_run':downloaded,'total_bytes':sum(s['bytes'] for s in manifest['shards']),
                 'elapsed_seconds':time.monotonic()-started})
    atomic_json(DEST/'DOWNLOAD_COMPLETE.json',{'verified_shards':len(complete),'revision':REV})

if __name__=='__main__':main()
