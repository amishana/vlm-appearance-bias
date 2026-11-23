"""Fresh linear/MLP probes; choose epochs on disjoint dev groups, never val."""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch


def balanced(y, pred):
    assert set(np.unique(y)) == {0, 1}
    return float(np.mean([(pred[y == c] == c).mean() for c in (0,1)]))


def fit_probe(fit, dev, val, kind, seed, epochs=60):
    torch.manual_seed(seed)
    mean = fit['x'].mean(axis=0)
    std = fit['x'].std(axis=0).clip(1e-5)
    xs = [torch.tensor((d['x']-mean)/std, dtype=torch.float32) for d in (fit, dev, val)]
    ys = [torch.tensor(d['y'], dtype=torch.long) for d in (fit, dev, val)]
    dim = xs[0].shape[1]
    model = torch.nn.Linear(dim,2) if kind == 'linear' else torch.nn.Sequential(
        torch.nn.Linear(dim,256), torch.nn.ReLU(), torch.nn.Linear(256,2))
    counts = torch.bincount(ys[0], minlength=2).float()
    weights = counts.sum()/(2*counts)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    learning_rate = 1e-2 if kind == 'linear' else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    best, state, best_epoch, stale = -1, None, 0, 0
    for epoch in range(1,epochs+1):
        model.train()
        for idx in torch.randperm(len(xs[0])).split(256):
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xs[0][idx]),ys[0][idx])
            assert torch.isfinite(loss).item()
            loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            score = balanced(dev['y'],model(xs[1]).argmax(1).numpy())
        if score > best + 1e-6:
            best, state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch >= 30 and stale >= 15:
            break
    model.load_state_dict(state)
    with torch.inference_mode():
        pred = model(xs[2]).argmax(1).numpy()
    return dict(best_epoch=best_epoch, learning_rate=learning_rate, dev_balanced_accuracy=best,
                val_accuracy=float((pred == val['y']).mean()),
                val_balanced_accuracy=balanced(val['y'],pred)), pred, dict(
                    state=state, mean=torch.tensor(mean), std=torch.tensor(std), kind=kind, seed=seed)


def bootstrap(y, pred, groups, seed=566):
    unique, inv = np.unique(groups,return_inverse=True)
    totals=np.zeros((len(unique),2)); hits=totals.copy()
    for c in (0,1):
        totals[:,c]=np.bincount(inv,weights=(y==c),minlength=len(unique))
        hits[:,c]=np.bincount(inv,weights=((y==c)&(pred==c)),minlength=len(unique))
    rng=np.random.default_rng(seed); values=[]
    for _ in range(1000):
        idx=rng.integers(0,len(unique),len(unique)); n=totals[idx].sum(0)
        if (n>0).all():values.append((hits[idx].sum(0)/n).mean())
    return np.percentile(values,[2.5,97.5]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    assert (a.features/'SUCCESS').exists()
    results=[];reference={}
    for condition in ('original','ce','adv'):
        data={}
        for split in ('fit','dev','val'):
            with np.load(a.features/f'{condition}_{split}.npz',allow_pickle=False) as z:
                data[split]={k:z[k] for k in z.files}
            d=data[split];assert np.isfinite(d['x']).all()
            assert set(np.unique(d['y']))=={0,1}
            if condition=='original':reference[split]=(d['ids'],d['y'],d['groups'])
            else:
                for got,expected in zip((d['ids'],d['y'],d['groups']),reference[split]):assert np.array_equal(got,expected)
        sets=[set(data[s]['groups']) for s in ('fit','dev','val')]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        for kind in ('linear','mlp'):
            for seed in (42,73):
                result,pred,checkpoint=fit_probe(data['fit'],data['dev'],data['val'],kind,seed)
                low,high=bootstrap(data['val']['y'],pred,data['val']['groups'])
                result.update(condition=condition,probe=kind,seed=seed,
                              val_balanced_ci_low=low,val_balanced_ci_high=high,
                              n_fit=len(data['fit']['y']),n_dev=len(data['dev']['y']),n_val=len(data['val']['y']))
                results.append(result)
                stem=f'{condition}_{kind}_{seed}'
                torch.save(checkpoint,a.out/f'{stem}.pt')
                with (a.out/f'{stem}_predictions.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['image_id','counterfactual_set','gender_label','predicted_gender'])
                    w.writerows(zip(data['val']['ids'],data['val']['groups'],data['val']['y'],pred))
                (a.out/'summary.json').write_text(json.dumps(results,indent=2))
                print(result,flush=True)
    with (a.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (a.out/'SUCCESS').write_text('Probes completed\n')


if __name__=='__main__':main()
