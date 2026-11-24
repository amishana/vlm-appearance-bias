# Mitigation vs `smolvlm2`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## smolvlm2_P1_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 58.9% | 57.5% | -1.3 pp | [-2.5 pp, -0.2 pp] |
| accuracy_category |  | 79.9% | 79.2% | -0.7 pp | [-1.9 pp, +0.9 pp] |
| macro_f1 |  | 45.2% | 45.0% | -0.2 pp |  |
| attribute_gap | gender | 0.2% | 0.2% | -0.0 pp | [-1.6 pp, +1.5 pp] |
| flip_rate | gender | 23.5% | 24.7% | +1.2 pp | [-1.1 pp, +3.7 pp] |
| attribute_gap | race | 1.7% | 4.5% | +2.8 pp | [-0.7 pp, +5.8 pp] |
| flip_rate | race | 15.0% | 14.8% | -0.2 pp | [-2.4 pp, +2.0 pp] |
| attribute_gap | physical | 3.7% | 3.7% | -0.0 pp | [-2.9 pp, +4.1 pp] |
| flip_rate | physical | 21.1% | 23.6% | +2.6 pp | [+0.2 pp, +4.7 pp] |
| consistent_sets |  | 54.8% | 51.3% | -3.5 pp | [-7.0 pp, +0.0 pp] |
| mean_distinct |  | 2.005 | 1.970 | -0.035 | [-0.146, +0.065] |
| changed_predictions |  |  | 10.1% | +10.1 pp | [+8.0 pp, +12.3 pp] |

McNemar (accuracy): baseline right / method wrong = 68, baseline wrong / method right = 35, exact p = 0.00149.

## smolvlm2_P2_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 58.9% | 63.5% | +4.6 pp | [+1.1 pp, +8.0 pp] |
| accuracy_category |  | 79.9% | 83.4% | +3.5 pp | [+0.4 pp, +6.6 pp] |
| macro_f1 |  | 45.2% | 52.1% | +7.0 pp |  |
| attribute_gap | gender | 0.2% | 1.5% | +1.3 pp | [-2.6 pp, +3.6 pp] |
| flip_rate | gender | 23.5% | 20.9% | -2.6 pp | [-6.0 pp, +0.8 pp] |
| attribute_gap | race | 1.7% | 2.4% | +0.7 pp | [-2.7 pp, +3.3 pp] |
| flip_rate | race | 15.0% | 13.6% | -1.4 pp | [-4.7 pp, +2.4 pp] |
| attribute_gap | physical | 3.7% | 4.1% | +0.5 pp | [-5.3 pp, +5.4 pp] |
| flip_rate | physical | 21.1% | 22.1% | +1.1 pp | [-3.9 pp, +6.1 pp] |
| consistent_sets |  | 54.8% | 56.8% | +2.0 pp | [-4.0 pp, +7.5 pp] |
| mean_distinct |  | 2.005 | 1.940 | -0.065 | [-0.211, +0.085] |
| changed_predictions |  |  | 24.7% | +24.7 pp | [+19.9 pp, +29.8 pp] |

McNemar (accuracy): baseline right / method wrong = 91, baseline wrong / method right = 204, exact p = 4.07e-11.

## smolvlm2_P4_sub200dbg

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 58.9% | 63.4% | +4.6 pp | [+1.3 pp, +7.8 pp] |
| accuracy_category |  | 79.9% | 83.9% | +4.1 pp | [+1.2 pp, +7.1 pp] |
| macro_f1 |  | 45.2% | 53.3% | +8.1 pp |  |
| attribute_gap | gender | 0.2% | 1.0% | +0.8 pp | [-2.6 pp, +2.5 pp] |
| flip_rate | gender | 23.5% | 21.4% | -2.0 pp | [-5.3 pp, +1.3 pp] |
| attribute_gap | race | 1.7% | 2.1% | +0.3 pp | [-2.8 pp, +3.6 pp] |
| flip_rate | race | 15.0% | 14.6% | -0.4 pp | [-3.6 pp, +2.8 pp] |
| attribute_gap | physical | 3.7% | 6.9% | +3.2 pp | [-3.4 pp, +7.0 pp] |
| flip_rate | physical | 21.1% | 21.4% | +0.4 pp | [-3.8 pp, +4.5 pp] |
| consistent_sets |  | 54.8% | 55.8% | +1.0 pp | [-5.0 pp, +7.0 pp] |
| mean_distinct |  | 2.005 | 1.945 | -0.060 | [-0.201, +0.075] |
| changed_predictions |  |  | 24.6% | +24.6 pp | [+19.9 pp, +29.8 pp] |

McNemar (accuracy): baseline right / method wrong = 75, baseline wrong / method right = 187, exact p = 3.28e-12.
