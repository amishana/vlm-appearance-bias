# Mitigation vs `qwen2_5_vl`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## qwen2_5_vl_P1_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 57.7% | 58.8% | +1.1 pp | [-0.1 pp, +2.4 pp] |
| accuracy_category |  | 78.9% | 80.9% | +2.0 pp | [+0.7 pp, +3.4 pp] |
| macro_f1 |  | 45.9% | 44.7% | -1.2 pp |  |
| attribute_gap | gender | 2.3% | 1.4% | -0.9 pp | [-2.5 pp, +0.9 pp] |
| flip_rate | gender | 19.9% | 18.4% | -1.5 pp | [-3.7 pp, +0.7 pp] |
| attribute_gap | race | 2.8% | 2.1% | -0.7 pp | [-4.0 pp, +2.2 pp] |
| flip_rate | race | 13.4% | 12.8% | -0.6 pp | [-2.7 pp, +1.4 pp] |
| attribute_gap | physical | 3.7% | 3.2% | -0.5 pp | [-4.5 pp, +3.5 pp] |
| flip_rate | physical | 20.6% | 17.6% | -3.1 pp | [-6.0 pp, -0.5 pp] |
| consistent_sets |  | 55.3% | 56.8% | +1.5 pp | [-2.5 pp, +5.5 pp] |
| mean_distinct |  | 1.910 | 1.824 | -0.085 | [-0.176, +0.000] |
| changed_predictions |  |  | 7.6% | +7.6 pp | [+5.7 pp, +9.5 pp] |

McNemar (accuracy): baseline right / method wrong = 28, baseline wrong / method right = 55, exact p = 0.00404.

## qwen2_5_vl_P2_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 57.7% | 58.7% | +0.9 pp | [-1.5 pp, +3.3 pp] |
| accuracy_category |  | 78.9% | 82.3% | +3.4 pp | [+0.4 pp, +6.8 pp] |
| macro_f1 |  | 45.9% | 48.0% | +2.1 pp |  |
| attribute_gap | gender | 2.3% | 2.6% | +0.3 pp | [-1.8 pp, +2.2 pp] |
| flip_rate | gender | 19.9% | 22.2% | +2.3 pp | [-1.1 pp, +5.6 pp] |
| attribute_gap | race | 2.8% | 4.5% | +1.7 pp | [-3.0 pp, +4.0 pp] |
| flip_rate | race | 13.4% | 16.2% | +2.8 pp | [-0.1 pp, +5.7 pp] |
| attribute_gap | physical | 3.7% | 3.2% | -0.5 pp | [-6.9 pp, +6.7 pp] |
| flip_rate | physical | 20.6% | 21.0% | +0.3 pp | [-4.9 pp, +5.7 pp] |
| consistent_sets |  | 55.3% | 57.3% | +2.0 pp | [-4.0 pp, +8.0 pp] |
| mean_distinct |  | 1.910 | 2.040 | +0.131 | [-0.020, +0.276] |
| changed_predictions |  |  | 20.0% | +20.0 pp | [+16.0 pp, +24.1 pp] |

McNemar (accuracy): baseline right / method wrong = 89, baseline wrong / method right = 112, exact p = 0.121.

## qwen2_5_vl_P4_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 57.7% | 59.1% | +1.4 pp | [-1.0 pp, +3.8 pp] |
| accuracy_category |  | 78.9% | 82.1% | +3.2 pp | [+0.2 pp, +6.8 pp] |
| macro_f1 |  | 45.9% | 47.5% | +1.6 pp |  |
| attribute_gap | gender | 2.3% | 2.0% | -0.3 pp | [-2.9 pp, +2.0 pp] |
| flip_rate | gender | 19.9% | 21.7% | +1.8 pp | [-1.2 pp, +4.7 pp] |
| attribute_gap | race | 2.8% | 3.8% | +1.0 pp | [-2.6 pp, +3.4 pp] |
| flip_rate | race | 13.4% | 15.8% | +2.5 pp | [-0.9 pp, +5.7 pp] |
| attribute_gap | physical | 3.7% | 4.1% | +0.5 pp | [-7.4 pp, +6.9 pp] |
| flip_rate | physical | 20.6% | 18.4% | -2.2 pp | [-7.2 pp, +3.1 pp] |
| consistent_sets |  | 55.3% | 54.3% | -1.0 pp | [-7.5 pp, +5.5 pp] |
| mean_distinct |  | 1.910 | 1.975 | +0.065 | [-0.080, +0.206] |
| changed_predictions |  |  | 20.7% | +20.7 pp | [+16.5 pp, +25.0 pp] |

McNemar (accuracy): baseline right / method wrong = 89, baseline wrong / method right = 123, exact p = 0.0232.
