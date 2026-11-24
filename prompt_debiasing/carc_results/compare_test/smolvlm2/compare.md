# Mitigation vs `smolvlm2_P0_sub200test`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## smolvlm2_P2_sub200test

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 60.3% | 66.1% | +5.9 pp | [+3.0 pp, +8.8 pp] |
| accuracy_category |  | 80.3% | 83.3% | +2.9 pp | [+0.7 pp, +5.0 pp] |
| macro_f1 |  | 48.0% | 56.6% | +8.6 pp |  |
| attribute_gap | gender | 3.0% | 1.9% | -1.1 pp | [-4.4 pp, +2.2 pp] |
| flip_rate | gender | 20.1% | 19.9% | -0.2 pp | [-3.8 pp, +3.5 pp] |
| attribute_gap | race | 1.7% | 3.1% | +1.4 pp | [-2.7 pp, +4.2 pp] |
| flip_rate | race | 13.2% | 12.9% | -0.3 pp | [-3.1 pp, +2.4 pp] |
| attribute_gap | physical | 5.2% | 8.1% | +2.9 pp | [-3.2 pp, +8.6 pp] |
| flip_rate | physical | 16.5% | 17.6% | +1.1 pp | [-2.7 pp, +5.1 pp] |
| consistent_sets |  | 58.3% | 58.3% | +0.0 pp | [-6.0 pp, +5.5 pp] |
| mean_distinct |  | 1.809 | 1.859 | +0.050 | [-0.080, +0.196] |
| changed_predictions |  |  | 19.9% | +19.9 pp | [+16.2 pp, +24.0 pp] |

McNemar (accuracy): baseline right / method wrong = 69, baseline wrong / method right = 212, exact p = 4.41e-18.
