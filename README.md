# GroupNormalization-tensorflow
a Tensorflow implementation of Group Normalizations  proposed in the paper [Group Normalization](https://arxiv.org/abs/1803.08494) by Wu et al.

# Description
- based on the GN implementation by [shaohua0116](https://github.com/shaohua0116/Group-Normalization-Tensorflow)
- add exponential moving average process about global mean & variance
- in different condition, the group normalization behave is different like batch normalization. when train step, it need to update ema_mean & ema_var every step. however, in inference step, it need to get ema_mean & ema_var that computed by trainning step
