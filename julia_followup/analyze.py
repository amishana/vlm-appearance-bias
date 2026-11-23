"""Apply the repository's frozen canonical scorer to matched validation CSVs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from eval.score import score_frame
from eval import metrics


def paired_ci(a,b,seed=566):
    """Paired group bootstrap for image accuracy and full-group consistency."""
    assert a['image_id'].tolist()==b['image_id'].tolist()
    agg={}
    for name,df in [('a',a),('b',b)]:
        agg[name]=df.groupby('counterfactual_set',sort=True).agg(
            correct=('correct_synonym','sum'),n=('image_id','size'),distinct=('pred_canonical','nunique'))
    x,y=agg['a'],agg['b'];assert x.index.equals(y.index)
    assert np.array_equal(x['n'],y['n'])
    dc=(y['correct']-x['correct']).to_numpy();n=x['n'].to_numpy()
    ds=((y['distinct']==1).astype(float)-(x['distinct']==1).astype(float)).to_numpy()
    rng=np.random.default_rng(seed);acc=[];cons=[]
    for _ in range(1000):
        idx=rng.integers(0,len(n),len(n));acc.append(float(dc[idx].sum()/n[idx].sum()));cons.append(float(ds[idx].mean()))
    return dict(accuracy_delta=float(dc.sum()/n.sum()),accuracy_ci=np.percentile(acc,[2.5,97.5]).tolist(),
                consistency_delta=float(ds.mean()),consistency_ci=np.percentile(cons,[2.5,97.5]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--extended',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/scratch2/jlchun/csci566/processed/socialcounterfactuals/exported'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    val=pd.read_csv(a.data/'val.csv',dtype=str,keep_default_na=False)
    assert val['image_id'].is_unique
    expected=val.set_index('image_id');frames={};reference=None
    for condition in ('original','ce','adv'):
        path=a.extended/'evaluate/evaluation'/f'{condition}_predictions.csv'
        df=pd.read_csv(path,dtype=str,keep_default_na=False).sort_values('image_id').reset_index(drop=True)
        assert len(df)>0 and df['image_id'].is_unique
        if reference is None:reference=df['image_id'].tolist()
        else:assert reference==df['image_id'].tolist(),'Conditions evaluated on different images'
        selected=set(df['counterfactual_set'])
        assert set(df['image_id'])==set(val.loc[val['counterfactual_set'].isin(selected),'image_id']),'Incomplete validation groups'
        gold=expected.loc[df['image_id']].reset_index()
        for col in ('occupation_raw','counterfactual_set','dataset_segment','a1_type','a1','a2_type','a2'):
            assert df[col].tolist()==gold[col].tolist(),f'Metadata mismatch: {col}'
        assert df['model_response'].str.strip().ne('').all()
        scored=score_frame(df)
        assert np.array_equal(scored['correct_strict'].to_numpy(),df['correct'].astype(int).to_numpy())
        folder=a.out/'scored'/condition;folder.mkdir(parents=True)
        scored.to_csv(folder/'val_scored.csv',index=False)
        frames[condition]=scored.loc[scored['excluded']==0].reset_index(drop=True)
    metrics.main(['--scored',str(a.out/'scored'),'--out',str(a.out/'metrics')])
    paired={f'{b}_minus_{c}':paired_ci(frames[c],frames[b]) for c,b in [('original','ce'),('original','adv'),('ce','adv')]}
    (a.out/'paired_comparisons.json').write_text(json.dumps(paired,indent=2))
    # A simple standalone SVG without introducing plotting dependencies.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="330" viewBox="0 0 820 330">',
           '<rect width="820" height="330" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#172033}</style>']
    for panel,(title,fn) in enumerate([
        ('Canonical occupation accuracy',lambda d:float(d['correct_synonym'].mean())),
        ('Full-group consistency',lambda d:float((d.groupby('counterfactual_set')['pred_canonical'].nunique()==1).mean()))]):
        x0=30+panel*410
        parts.append(f'<text x="{x0}" y="28">{title}</text>')
        for i,(condition,color) in enumerate([('original','#64748b'),('ce','#2563eb'),('adv','#0f766e')]):
            value=fn(frames[condition]);h=value*200;x=x0+25+i*115
            parts.extend([f'<rect x="{x}" y="{260-h}" width="70" height="{h}" fill="{color}"/>',
                          f'<text x="{x}" y="{250-h}">{value:.1%}</text>',f'<text x="{x}" y="285">{condition}</text>'])
    parts.append('<text x="30" y="315">Validation subset; canonical exclusions applied. See paired bootstrap intervals in JSON.</text></svg>')
    (a.out/'comparison.svg').write_text('\n'.join(parts))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve().parents[1]/'eval').glob('*.py')}
    (a.out/'scorer_hashes.json').write_text(json.dumps(hashes,indent=2))
    (a.out/'SUCCESS').write_text('Canonical analysis completed\n')
    print(json.dumps(paired,indent=2),flush=True)


if __name__=='__main__':main()
