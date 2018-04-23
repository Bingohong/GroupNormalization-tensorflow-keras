# GroupNormalization-tensorflow
a Tensorflow implementation of Group Normalizations  proposed in the paper [Group Normalization](https://arxiv.org/abs/1803.08494) by Wu et al.

# Description
- based on the GN implementation by [shaohua0116](https://github.com/shaohua0116/Group-Normalization-Tensorflow)
- add exponential moving average process about global mean & variance
- in batch normalization, the mean/variance is different. 
  - when train step, it need to update ema_mean & ema_var every step. 
  - however, in inference step, it need to get ema_mean & ema_var that computed by trainning step
- For group normalization, does it need to record moving average of mean/variance? So, I compare 3 condition normalization:
  - batch normalizaiton by keras in-built layers [Normalization](https://keras.io/zh/layers/normalization/)
  - group normalization without moving average by [shaohua0116](https://github.com/shaohua0116/Group-Normalization-Tensorflow)
  - group normalization with moving average by [here](https://github.com/Bingohong/GroupNormalization-tensorflow/blob/master/group_normalization_keras.py)
- compare with batch_normalization and instance_normalization, the IN doesn't need to record moving average.
  
