#!/usr/bin/env python3
"""Run original benchmark scripts on verified cached shards, preserving row order."""
import json
from pathlib import Path
import runpy
import sys
import datasets

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/benchmark_parquet_20261027'
manifest = json.loads((CACHE/'source_manifest.json').read_text())
if not (CACHE/'DOWNLOAD_COMPLETE.json').exists():
    raise RuntimeError('Pinned shard download is incomplete')
original_loader = datasets.load_dataset


def local_loader(name, *args, **kwargs):
    if name != manifest['dataset'] or kwargs.get('revision') != manifest['revision']:
        raise ValueError('Dataset/revision differs from verified local cache')
    return original_loader('parquet', data_files={'train':[str(CACHE/s['path']) for s in manifest['shards']]},
                           split='train', streaming=True)


datasets.load_dataset = local_loader
script = sys.argv[1]
if script not in {'preprocess_dataset','export_images'}:
    raise ValueError('Unsupported benchmark script')
sys.argv = [script+'.py'] + sys.argv[2:]
sys.path.insert(0,str(ROOT/'benchmark_source/data'))
runpy.run_path(str(ROOT/'benchmark_source/data'/f'{script}.py'),run_name='__main__')
