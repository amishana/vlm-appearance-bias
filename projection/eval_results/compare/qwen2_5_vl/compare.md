# Mitigation vs `qwen2_5_vl_local`

Delta = method minus baseline on the same images; 95% CI from a paired bootstrap over counterfactual sets. Lower is better for gap, flip rate and mean distinct; higher is better for accuracy, macro F1 and consistent sets.


## qwen2_5_vl_leace_all

| metric | attribute | baseline | method | delta | 95% CI |
|---|---|---|---|---|---|
| accuracy |  | 56.5% | 57.0% | +0.6 pp | [+0.3 pp, +0.8 pp] |
| accuracy_category |  | 77.3% | 77.7% | +0.4 pp | [+0.1 pp, +0.7 pp] |
| macro_f1 |  | 45.0% | 45.2% | +0.2 pp |  |
| attribute_gap | gender | 0.3% | 0.1% | -0.2 pp | [-0.7 pp, +0.6 pp] |
| flip_rate | gender | 22.2% | 18.4% | -3.7 pp | [-4.4 pp, -3.1 pp] |
| attribute_gap | race | 4.6% | 3.8% | -0.8 pp | [-1.5 pp, +0.0 pp] |
| flip_rate | race | 16.4% | 15.6% | -0.7 pp | [-1.2 pp, -0.3 pp] |
| attribute_gap | physical | 1.7% | 1.6% | -0.1 pp | [-1.0 pp, +0.9 pp] |
| flip_rate | physical | 21.3% | 19.7% | -1.6 pp | [-2.5 pp, -0.8 pp] |
| consistent_sets |  | 53.0% | 56.4% | +3.4 pp | [+2.3 pp, +4.5 pp] |
| mean_distinct |  | 1.937 | 1.790 | -0.147 | [-0.170, -0.123] |
| changed_predictions |  |  | 6.6% | +6.6 pp | [+6.1 pp, +7.1 pp] |

McNemar (accuracy): baseline right / method wrong = 245, baseline wrong / method right = 392, exact p = 6.26e-09.
