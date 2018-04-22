import tensorflow as tf

def GroupNormalization(x, phase_train, groups=32, decay=0.99, scope="gn"):
	"""
	Group normalization layer

	Group Normalization divides the channels into groups and computes within each group
	the mean and variance for normalization. GN's computation is independent of batch sizes,
	and its accuracy is stable in a wide range of batch sizes

	Arguments
	---------
	x: tensor, 4D [batch_size, height, weight, channels] input maps
	phase_train:  boolean tf.Variable, true indicates training phase
	groups: Integer, the number of groups for Group Normalization.
	decay: control ExponentialMovingAverage
	scope:string, variable scope

	Output shape
	------------
	Same shape as input.

	Return
	------
	group-normalization maps

	ExponentialMovingAverage method
	-------------------------------
	- apply()方法添加了训练变量的影子副本，并保持了其影子副本中训练变量的移动平均值操作。在每次训练之后调用此操作，更新移动平均值。
	- average()和average_name()方法可以获取影子变量及其名称。
	- EMA 操作不会改变变量本身的值，而是维护一个影子变量记录其滑动平均值，当需要使用这个滑动平均值时可以使用average函数

	References
	----------
	- [Group Normalization](https://arxiv.org/abs/1803.08494)
	- https://github.com/shaohua0116/Group-Normalization-Tensorflow/blob/master/ops.py
	"""
	with tf.variable_scope(scope):
	    # normalize
	    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
	    esp = 1e-5
	    G = groups
	    x = tf.transpose(x, [0, 3, 1, 2])
	    N, C, H, W = x.get_shape().as_list()
	    G = min(G, C)
	    x = tf.reshape(x, [-1, G, C // G, H, W])

	    # per channel gamma and beta
	    gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))
	    beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))
	    gamma = tf.reshape(gamma, [1, C, 1, 1])
	    beta = tf.reshape(beta, [1, C, 1, 1])

	    # compute group-channel mean & variance
	    gn_mean, gn_var = tf.nn.moments(x, [2, 3, 4], keep_dims=True, name="moments")

	    # ema ops to record global mean & var update
	    ema = tf.train.ExponentialMovingAverage(decay=decay)
	    ema_mean, ema_var = ema.average(gn_mean), ema.average(gn_var)

	    def mean_var_with_update():
	    	# add shadow variable
	    	ema_apply_op = ema.apply([gn_mean,gn_var])
	    	with tf.control_dependencies([ema_apply_op]):
	    		return tf.identity(gn_mean), tf.identity(gn_var)

	    # depend on phase_train 
	    # when phase train is True, update shadow variable then return gn_mean & gn_var
	    # when phase train if False, return ema_mean & ema_var
	    mean, var = tf.cond(phase_train, mean_var_with_update, lambda:(ema_mean,ema_var))

	    # compute group normalization
	    x = (x - mean) / tf.sqrt(var + esp)
	    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

	    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
	    output = tf.transpose(output, [0, 2, 3, 1])
	return output

