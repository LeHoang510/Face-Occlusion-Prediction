# Training Results

## Results Table

| # | Run Name | Backbone | Best Val Score | Best Epoch | Total Epochs | Notes |
|---|---|---|---|---|---|---|
| 1 | swin_t | swin_t | **0.001090** | 15 | 20 | Best overall |
| 2 | convnext_tiny | convnext_tiny | 0.001112 | 4 | 20 | Suspicious early peak, lr too low |
| 3 | vit_b_16 | vit_b_16 | 0.001162 | 9 | 20 | Longest training time |
| 4 | convnext_tiny_v2 | convnext_tiny | 0.001169 | 8 | 20 | Rerun with lr=1e-4, worse than v1 |
| 5 | efficientnet_b3 | efficientnet_b3 | 0.001185 | 18 | 20 | Still improving at epoch 20 |
| 6 | twins_svt_small | twins_svt_small | 0.001207 | 14 | 20 | |
| 7 | resnet50 | resnet50 | 0.001221 | 12 | 20 | Baseline |
| 8 | dinov2_small | vit_small_patch14_dinov2 | 0.001469 | 19 | 20 | Bad + fluctuating |


The next step is to select the top three models for further improvement. The strategy should be:

1. Balanced batching

2. Segmentation as an auxiliary information

3. Multitask learning

4. Improve the prediction head deeper MLP heads, attention-based heads, or other lightweight architectural refinements

5. Perform ensemble learning using the best three models:

   * simple averaging ensemble
   * weighted averaging based on validation performance
   * stacking ensemble using a meta-learner (logistic regression or a small MLP) trained on the validation predictions of the selected models

6. After finding the best overall configuration, retrain the final model using the full training data (train + validation set), and evaluate it on the test set to obtain the final performance

