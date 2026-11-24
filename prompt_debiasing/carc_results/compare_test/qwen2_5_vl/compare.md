# Mitigation vs `qwen2_5_vl_P0_sub200test`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## qwen2_5_vl_P1_sub200test

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 61.8% | 61.9% | +0.2 pp | [-0.7 pp, +1.1 pp] |
| accuracy_category |  | 80.3% | 80.8% | +0.5 pp | [-0.3 pp, +1.4 pp] |
| macro_f1 |  | 51.1% | 50.5% | -0.6 pp |  |
| attribute_gap | gender | 0.9% | 0.7% | -0.2 pp | [-1.5 pp, +1.1 pp] |
| flip_rate | gender | 20.5% | 18.7% | -1.9 pp | [-3.8 pp, -0.2 pp] |
| attribute_gap | race | 3.1% | 3.8% | +0.7 pp | [-2.0 pp, +2.5 pp] |
| flip_rate | race | 13.6% | 13.2% | -0.4 pp | [-1.5 pp, +0.8 pp] |
| attribute_gap | physical | 3.8% | 3.8% | +0.0 pp | [-3.6 pp, +2.9 pp] |
| flip_rate | physical | 15.7% | 14.8% | -1.0 pp | [-3.1 pp, +1.1 pp] |
| consistent_sets |  | 58.8% | 61.8% | +3.0 pp | [+0.0 pp, +6.0 pp] |
| mean_distinct |  | 1.824 | 1.779 | -0.045 | [-0.116, +0.020] |
| changed_predictions |  |  | 5.7% | +5.7 pp | [+4.3 pp, +7.2 pp] |

McNemar (accuracy): baseline right / method wrong = 26, baseline wrong / method right = 30, exact p = 0.689.
