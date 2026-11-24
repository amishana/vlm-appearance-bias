# Mitigation vs `internvl3_P0_sub200test`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## internvl3_P1_sub200test

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 60.8% | 60.9% | +0.1 pp | [-1.9 pp, +2.0 pp] |
| accuracy_category |  | 80.3% | 80.0% | -0.4 pp | [-2.2 pp, +1.5 pp] |
| macro_f1 |  | 51.0% | 51.6% | +0.6 pp |  |
| attribute_gap | gender | 0.5% | 0.7% | +0.2 pp | [-2.6 pp, +2.1 pp] |
| flip_rate | gender | 20.3% | 19.0% | -1.2 pp | [-3.7 pp, +1.3 pp] |
| attribute_gap | race | 1.7% | 2.4% | +0.7 pp | [-3.5 pp, +3.5 pp] |
| flip_rate | race | 14.3% | 13.8% | -0.5 pp | [-2.4 pp, +1.3 pp] |
| attribute_gap | physical | 4.3% | 2.9% | -1.4 pp | [-5.6 pp, +3.4 pp] |
| flip_rate | physical | 15.3% | 13.6% | -1.7 pp | [-5.6 pp, +2.1 pp] |
| consistent_sets |  | 61.3% | 64.3% | +3.0 pp | [-1.5 pp, +7.5 pp] |
| mean_distinct |  | 1.834 | 1.714 | -0.121 | [-0.221, -0.020] |
| changed_predictions |  |  | 10.1% | +10.1 pp | [+7.7 pp, +12.6 pp] |

McNemar (accuracy): baseline right / method wrong = 65, baseline wrong / method right = 67, exact p = 0.931.
