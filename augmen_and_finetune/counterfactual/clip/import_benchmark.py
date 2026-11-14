#!/usr/bin/env python3
"""Import the team's exported CSVs without changing any split or sample ID."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
from prepare_data import normalize_row, validate_groups, audit_splits, SPLIT_NAMES


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def import_benchmark(exported, output):
    exported, output = Path(exported).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    splits, hashes = {}, {}
    for split in SPLIT_NAMES:
        source = exported / f'{split}.csv'
        hashes[split] = digest(source)
        with source.open(newline='', encoding='utf-8-sig') as f:
            raw = list(csv.DictReader(f))
        if not raw:
            raise ValueError(f'Empty benchmark split: {split}')
        records = []
        for row in raw:
            if row.get('split') and row['split'] != split:
                raise ValueError('CSV split column disagrees with filename')
            sample = row.get('image_id', '')
            if not sample or not row.get('image_path') or not row.get('occupation_raw'):
                raise ValueError('Missing image_id, image_path or occupation_raw')
            value = normalize_row(dict(row, a1a2=row.get('a1a2') or f"{row['a1']} {row['a2']}",
                                       occupation=row['occupation_raw'],
                                       source_index=int(row['source_row_index'])), 0)
            path = (exported / row['image_path']).resolve()
            if not path.is_relative_to(exported):
                raise ValueError('Image path escapes benchmark export directory')
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                pixels = hashlib.sha256(f'{rgb.size}:RGB:'.encode() + rgb.tobytes()).hexdigest()
            value = {key: value[key] for key in ('occupation', 'group_id', 'dataset_segment',
                     'a1_type', 'a2_type', 'a1', 'a2', 'a1a2', 'source_index')}
            value.update(sample_id=sample, image_path=os.path.relpath(path, output),
                         image_sha256=digest(path), pixel_sha256=pixels)
            records.append(value)
        validate_groups(records, official=True)
        splits[split] = records
    audit = audit_splits(splits)
    labels = sorted({r['occupation'] for rows in splits.values() for r in rows})
    missing = set(labels) - {r['occupation'] for r in splits['train']}
    if missing:
        raise ValueError(f'Candidate occupations absent from training: {sorted(missing)}')
    summary = {'status': 'TEAM_BENCHMARK_IMPORTED_NOT_TRAINED', 'split_strategy': 'unchanged team CSV assignments',
               'source_csv_sha256': hashes, 'labels': labels, 'audit': audit,
               'splits': {s: {'rows': len(rs), 'groups': len({r['group_id'] for r in rs})} for s, rs in splits.items()}}
    output.mkdir(parents=True)
    for split, rows in splits.items():
        for row in rows:
            row['label_id'] = labels.index(row['occupation'])
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exported', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(import_benchmark(args.exported, args.output), indent=2))
