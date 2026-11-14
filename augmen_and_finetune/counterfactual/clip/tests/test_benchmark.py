import csv
import json
from pathlib import Path
import pytest
from import_benchmark import import_benchmark
from evaluate_predictions import compare, metrics

ROOT = Path(__file__).resolve().parents[1]


def fixture_export(tmp_path):
    # Synthetic pixels in the exact team CSV schema; no research-data dependency.
    from PIL import Image
    root = tmp_path / 'exported'
    (root/'images').mkdir(parents=True)
    index = 0
    for split_index, split in enumerate(('train','val','test')):
        rows = []
        for occupation in ('nurse', 'plumber'):
            group = f'a_photo_of_physical_gender_{occupation}_{split_index}_123'
            for a1 in ('skinny', 'obese', 'young', 'old', 'tattooed'):
                for a2 in ('male', 'female'):
                    sample = f'sc_{index:06d}'
                    path = root/'images'/f'{sample}.png'
                    Image.new('RGB',(32,32),(index*3, index*2, index)).save(path)
                    rows.append(dict(image_id=sample, source_row_index=index,
                        counterfactual_set=group, occupation_raw=occupation,
                        caption=f'A photo of a {a1} {a2} {occupation}',
                        image_path='images/'+path.name, split=split,
                        dataset_segment='physical_gender',a1_type='physical',
                        a2_type='gender',a1=a1,a2=a2))
                    index += 1
        with (root/f'{split}.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return root


def test_import_preserves_assignments_and_ids(tmp_path):
    root = fixture_export(tmp_path)
    output = tmp_path/'imported'
    summary = import_benchmark(root, output)
    assert summary['audit']['group_leakage'] is False
    for split in ('train','val','test'):
        with (root/f'{split}.csv').open() as handle:
            expected = [row['image_id'] for row in csv.DictReader(handle)]
        actual = [json.loads(l)['sample_id'] for l in (output/f'{split}.jsonl').read_text().splitlines()]
        assert actual == expected
    with pytest.raises(FileExistsError):
        import_benchmark(root, output)


def test_import_rejects_wrong_split(tmp_path):
    root=fixture_export(tmp_path)
    path=root/'val.csv'
    path.write_text(path.read_text().replace(',val', ',train'))
    with pytest.raises(ValueError,match='split column'):
        import_benchmark(root,tmp_path/'out')


def test_metrics_known_outcome():
    rows=[dict(group_id='g',occupation='a',predicted_occupation=p,dataset_segment='s',a1='x',a2=str(i),a1a2=str(i)) for i,p in enumerate(['a','a','b'])]
    m=metrics(rows)
    assert m['accuracy']==pytest.approx(2/3)
    assert m['all_variants_agreement']==0
    assert m['group_mean_pairwise_agreement']==pytest.approx(1/3)


def test_compare_rejects_unpaired_sample(tmp_path):
    row = dict(sample_id='sample',occupation='nurse',predicted_occupation='nurse',
               class_order_json=json.dumps(['nurse','plumber']))
    paths=[]
    for name, sample in (('original','sample'),('altered','changed')):
        path=tmp_path/f'{name}.csv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader()
            writer.writerow(dict(row,sample_id=sample))
        paths.append(path)
    with pytest.raises(ValueError, match='Sample sets differ'):
        compare(*paths)


def test_training_recovers_from_atomic_checkpoint(tmp_path, monkeypatch):
    import torch
    import train_lora
    cfg=train_lora.create_smoke_config({},tmp_path/'resume_smoke')
    cfg['training'].update(checkpoint_steps=1,resume=True,groups_per_batch=1)
    original_save=torch.save
    crashed=False
    def interrupt_after_checkpoint(value,path,*args,**kwargs):
        nonlocal crashed
        original_save(value,path,*args,**kwargs)
        if not crashed:
            # Complete atomic rename, then simulate an interruption before the next update.
            Path(path).replace(Path(path).with_name('resume.pt'))
            crashed=True
            raise InterruptedError('simulated interruption')
    monkeypatch.setattr(torch,'save',interrupt_after_checkpoint)
    with pytest.raises(InterruptedError):train_lora.train(cfg)
    monkeypatch.setattr(torch,'save',original_save)
    result=train_lora.train(cfg)
    metadata=json.loads((result/'metadata.json').read_text())
    assert metadata['optimizer_steps']==4
    assert metadata['adapter_reload_check']['passed']
    rows=[json.loads(l) for l in (result.parent/'training_batches.jsonl').read_text().splitlines()]
    assert [r['global_step'] for r in rows]==[1,2,3,4]


def test_paired_bootstrap_identical_predictions_has_zero_delta():
    from evaluate_predictions import paired_group_bootstrap
    rows=[dict(sample_id=str(i),group_id=str(i//2),occupation='a',predicted_occupation='a') for i in range(8)]
    result=paired_group_bootstrap(rows,rows,repeats=100)
    assert result['independent_groups']==4
    assert all(value==[0.0,0.0] for value in result['delta_intervals'].values())


def test_cifar_labels_ignore_trailing_blank_line():
    from evaluate_capability import parse_labels
    raw=b'airplane\nautomobile\nbird\ncat\ndeer\ndog\nfrog\nhorse\nship\ntruck\n\n'
    assert len(parse_labels(raw))==10
    with pytest.raises(ValueError): parse_labels(b'a\nb\n')


def test_shell_runner_end_to_end_with_local_random_clip(tmp_path):
    import os
    import subprocess
    import sys
    from train_lora import create_smoke_config
    root=fixture_export(tmp_path)
    cfg=create_smoke_config({},tmp_path/'random_clip')
    env=dict(os.environ,PYTHON_BIN=sys.executable,CONFIG_PATH=cfg['_config_path'])
    result=subprocess.run(['bash',str(ROOT/'scripts/run_benchmark.sh'),str(root),str(tmp_path/'run')],
                          env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    comparison=json.loads((tmp_path/'run/comparison.json').read_text())
    assert comparison['baseline']['images']==20
    assert comparison['adapted']['images']==20
