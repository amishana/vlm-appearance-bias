# Mitigation vs `internvl3`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## internvl3_P1_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 58.7% | 58.8% | +0.1 pp | [-1.8 pp, +2.0 pp] |
| accuracy_category |  | 81.2% | 79.9% | -1.3 pp | [-3.2 pp, +0.4 pp] |
| macro_f1 |  | 46.4% | 46.8% | +0.3 pp |  |
| attribute_gap | gender | 0.1% | 1.0% | +1.0 pp | [-1.5 pp, +2.6 pp] |
| flip_rate | gender | 18.3% | 16.8% | -1.5 pp | [-4.0 pp, +0.9 pp] |
| attribute_gap | race | 3.1% | 3.1% | +0.0 pp | [-3.2 pp, +2.9 pp] |
| flip_rate | race | 12.2% | 10.8% | -1.4 pp | [-4.2 pp, +1.1 pp] |
| attribute_gap | physical | 4.1% | 5.0% | +0.9 pp | [-5.6 pp, +6.8 pp] |
| flip_rate | physical | 16.3% | 14.8% | -1.5 pp | [-4.8 pp, +1.8 pp] |
| consistent_sets |  | 63.8% | 62.8% | -1.0 pp | [-6.0 pp, +3.5 pp] |
| mean_distinct |  | 1.809 | 1.693 | -0.116 | [-0.216, -0.025] |
| changed_predictions |  |  | 11.2% | +11.2 pp | [+8.2 pp, +14.5 pp] |

McNemar (accuracy): baseline right / method wrong = 52, baseline wrong / method right = 54, exact p = 0.923.

## internvl3_P2_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 58.7% | 55.4% | -3.3 pp | [-6.6 pp, -0.2 pp] |
| accuracy_category |  | 81.2% | 75.5% | -5.7 pp | [-8.6 pp, -2.9 pp] |
| macro_f1 |  | 46.4% | 48.8% | +2.4 pp |  |
| attribute_gap | gender | 0.1% | 0.5% | +0.4 pp | [-1.7 pp, +2.8 pp] |
| flip_rate | gender | 18.3% | 30.7% | +12.4 pp | [+8.6 pp, +16.3 pp] |
| attribute_gap | race | 3.1% | 5.6% | +2.4 pp | [-2.5 pp, +7.1 pp] |
| flip_rate | race | 12.2% | 25.6% | +13.4 pp | [+9.5 pp, +17.4 pp] |
| attribute_gap | physical | 4.1% | 2.8% | -1.4 pp | [-6.5 pp, +8.4 pp] |
| flip_rate | physical | 16.3% | 30.1% | +13.8 pp | [+8.7 pp, +19.7 pp] |
| consistent_sets |  | 63.8% | 43.2% | -20.6 pp | [-27.6 pp, -14.6 pp] |
| mean_distinct |  | 1.809 | 2.628 | +0.819 | [+0.623, +1.035] |
| changed_predictions |  |  | 27.6% | +27.6 pp | [+22.7 pp, +32.7 pp] |

McNemar (accuracy): baseline right / method wrong = 176, baseline wrong / method right = 94, exact p = 6.83e-07.

## internvl3_P4_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 58.7% | 53.7% | -5.0 pp | [-8.7 pp, -1.5 pp] |
| accuracy_category |  | 81.2% | 72.6% | -8.5 pp | [-12.2 pp, -5.0 pp] |
| macro_f1 |  | 46.4% | 47.6% | +1.2 pp |  |
| attribute_gap | gender | 0.1% | 1.8% | +1.7 pp | [-1.9 pp, +3.5 pp] |
| flip_rate | gender | 18.3% | 25.7% | +7.4 pp | [+3.3 pp, +11.6 pp] |
| attribute_gap | race | 3.1% | 2.1% | -1.0 pp | [-3.4 pp, +3.7 pp] |
| flip_rate | race | 12.2% | 20.8% | +8.6 pp | [+5.5 pp, +12.0 pp] |
| attribute_gap | physical | 4.1% | 2.3% | -1.8 pp | [-7.1 pp, +7.6 pp] |
| flip_rate | physical | 16.3% | 27.0% | +10.7 pp | [+6.2 pp, +15.4 pp] |
| consistent_sets |  | 63.8% | 48.2% | -15.6 pp | [-22.1 pp, -9.0 pp] |
| mean_distinct |  | 1.809 | 2.302 | +0.492 | [+0.317, +0.673] |
| changed_predictions |  |  | 26.5% | +26.5 pp | [+21.8 pp, +31.5 pp] |

McNemar (accuracy): baseline right / method wrong = 194, baseline wrong / method right = 71, exact p = 2.25e-14.
