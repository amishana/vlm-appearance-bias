#!/usr/bin/env python3
"""Paired occupation evaluation; subgroup gaps are descriptive, not causal."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_predictions(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty predictions or duplicate sample IDs')
    for row in rows:
        labels = json.loads(row['class_order_json'])
        if row['occupation'] not in labels or row['predicted_occupation'] not in labels:
            raise ValueError('Prediction outside candidate vocabulary')
        if labels != json.loads(rows[0]['class_order_json']):
            raise ValueError('Inconsistent candidate vocabulary')
    return rows


def metrics(rows):
    def accuracy(values):
        return sum(r['occupation'] == r['predicted_occupation'] for r in values) / len(values)
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    if any(len(rs) < 2 or len({r['occupation'] for r in rs}) != 1 for rs in groups.values()):
        raise ValueError('Incomplete or mixed-label counterfactual groups')
    pairs = []
    for values in groups.values():
        n = len(values)
        pairs.append(sum(a['predicted_occupation'] == b['predicted_occupation']
                         for i, a in enumerate(values) for b in values[i+1:]) / (n*(n-1)/2))
    breakdown = {}
    for attribute in ('a1', 'a2', 'a1a2'):
        cells = defaultdict(list)
        for row in rows:
            cells[(row['dataset_segment'], row[attribute])].append(row)
        breakdown[attribute] = {f'{segment}/{value}': {'n': len(rs), 'accuracy': accuracy(rs)}
                                for (segment, value), rs in sorted(cells.items())}
    gaps = {}
    for segment in sorted({r['dataset_segment'] for r in rows}):
        scores = [v['accuracy'] for k,v in breakdown['a1a2'].items() if k.startswith(segment + '/')]
        gaps[segment] = max(scores) - min(scores)
    return {'images': len(rows), 'groups': len(groups), 'accuracy': accuracy(rows),
            'group_balanced_accuracy': sum(accuracy(rs) for rs in groups.values()) / len(groups),
            'all_variants_agreement': sum(len({r['predicted_occupation'] for r in rs}) == 1 for rs in groups.values()) / len(groups),
            'group_mean_pairwise_agreement': sum(pairs)/len(pairs),
            'intersection_accuracy_gap_by_segment': gaps, 'subgroups': breakdown}


def paired_group_bootstrap(left, right, repeats=2000):
    """Resample whole matched groups, retaining unequal group sizes and pairing."""
    grouped = defaultdict(list)
    right = {r['sample_id']: r for r in right}
    for row in left:
        grouped[row['group_id']].append((row, right[row['sample_id']]))
    values = []
    for _, pairs in sorted(grouped.items()):
        n = len(pairs)
        correct_delta = sum((b['predicted_occupation'] == b['occupation']) -
                            (a['predicted_occupation'] == a['occupation']) for a,b in pairs)
        agreement = []
        unanimous = []
        for column in (0,1):
            predictions = [pair[column]['predicted_occupation'] for pair in pairs]
            unanimous.append(float(len(set(predictions)) == 1))
            agreement.append(sum(a == b for i,a in enumerate(predictions) for b in predictions[i+1:])/(n*(n-1)/2))
        values.append([correct_delta,n,correct_delta/n,unanimous[1]-unanimous[0],agreement[1]-agreement[0]])
    if len(values) < 2:
        return {'method':'Insufficient independent groups for bootstrap intervals'}
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(repeats):
        selected = values[rng.integers(len(values),size=len(values))]
        deltas.append([selected[:,0].sum()/selected[:,1].sum(),*selected[:,2:].mean(axis=0)])
    intervals = np.quantile(deltas,[0.025,0.975],axis=0)
    return {'method':'Paired whole-counterfactual-group bootstrap, percentile 95% intervals',
            'resamples':repeats,'seed':42,'independent_groups':len(values),
            'delta_intervals':{key:intervals[:,i].tolist() for i,key in enumerate(
                ('accuracy','group_balanced_accuracy','all_variants_agreement','group_mean_pairwise_agreement'))}}


def compare(baseline, adapted):
    left, right = read_predictions(baseline), read_predictions(adapted)
    by_id = {r['sample_id']: r for r in right}
    if {r['sample_id'] for r in left} != set(by_id):
        raise ValueError('Sample sets differ')
    for row in left:
        other = by_id[row['sample_id']]
        for key in ('group_id', 'occupation', 'dataset_segment', 'a1_type', 'a2_type', 'a1', 'a2',
                    'image_sha256', 'split', 'class_order_json', 'model_id'):
            if row[key] != other[key]:
                raise ValueError(f'Paired prediction mismatch: {key}')
    # Sidecars pin the model revision, prompt, and exact manifests.
    meta = [json.loads(Path(p).with_suffix('.metadata.json').read_text()) for p in (baseline, adapted)]
    if meta[0].get('adapter') or not meta[1].get('adapter'):
        raise ValueError('Expected original-model baseline and an adapted-model comparison')
    for path, metadata in zip((baseline, adapted), meta):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != metadata['output_sha256']:
            raise ValueError('Prediction file differs from recorded hash')
    if not meta[1].get('data_matches_training_manifests'):
        raise ValueError('Adapted evaluation input is not the training run held-out manifest')
    if any(m.get('split') != 'test' for m in meta):
        raise ValueError('Comparison requires held-out test predictions')
    for key in ('prompt_template', 'revision', 'base_model', 'split_audit'):
        if meta[0][key] != meta[1][key]:
            raise ValueError(f'Prediction provenance differs: {key}')
    a, b = metrics(left), metrics(right)
    return {'baseline': a, 'adapted': b,
            'delta_adapted_minus_baseline': {k: b[k]-a[k] for k in ('accuracy', 'group_balanced_accuracy', 'all_variants_agreement', 'group_mean_pairwise_agreement')},
            'intersection_gap_delta_by_segment': {k:b['intersection_accuracy_gap_by_segment'][k]-v for k,v in a['intersection_accuracy_gap_by_segment'].items()},
            'uncertainty': paired_group_bootstrap(left,right),
            'limitations': ['Single training seed; bootstrap reflects test-group sampling only, not training-seed variation or causal fairness.',
                            'Occupation accuracy is not external general-capability retention.',
                            'Results apply to the recorded candidate vocabulary and dataset, not untested VLMs or real-world fairness.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--adapted', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.adapted)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['delta_adapted_minus_baseline'], indent=2))
