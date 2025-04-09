# import numpy as np
# import tensorflow as tf
# import sklearn.preprocessing as prep
# # from tensorflow.examples.tutorials.mnist import input_data
#
#
# def xavier_init(fan_in, fan_out, constant=1):
#     low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
#     high = constant * np.sqrt(6.0 / (fan_in + fan_out))
#     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
#
#
# class Autoencoder(object):
#     def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
#                  optimizer=tf.keras.optimizers.Adam(), scale=0.2):
#         self.n_input = n_input
#         self.n_hidden = n_hidden
#         self.transfer = transfer_function
#         self.scale = tf.placeholder(tf.float32)
#         self.training_scale = scale
#         network_weights = self._initialize_weights()
#         self.weights = network_weights
#
#         self.x = tf.placeholder(tf.float32, [None, self.n_input])
#         self.noisex = self.x+scale*tf.random_normal((n_input,))
#
#         self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
#                                                       self.weights['w1']), self.weights['b1']))
#
#
#         self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
#
#         self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
#         self.optimizer = optimizer.minimize(self.cost)
#         init = tf.global_variables_initializer()
#         self.sess = tf.Session()
#         self.sess.run(init)
#
#
#     def _initialize_weights(self):
#         all_weights = dict()
#         all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
#         all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
#         all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
#         all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
#         return all_weights
#
#
#     def partial_fit(self, X):
#         cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
#                                                                           self.scale: self.training_scale})
#         return cost
#
#     def before_loss(self, X):
#         cost = self.sess.run((self.cost), feed_dict={self.x: X,
#                                                      self.scale: self.training_scale})
#         return cost
#
#
#     def transform(self, X):
#         return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})
#
#     def generate(self, hidden=None):
#         if hidden is None:
#             # print(self.weights["b1"].shape)
#             hidden = np.random.normal(size=self.weights["b1"])
#             # hidden = np.random.normal(size=self.weights["b1"].shape)
#
#         return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
#
#     def reconstruct(self, X):
#         return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})
#
#
#     def getWeights(self):
#         return self.sess.run(self.weights['w1'])
#
#     def getBias(self):
#         return self.sess.run(self.weights['b1'])

import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class Autoencoder(tf.keras.Model):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.keras.optimizers.Adam(), scale=0.2):
        super(Autoencoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.training_scale = scale
        self.optimizer = optimizer

        # 初始化权重
        self.w1 = tf.Variable(xavier_init(n_input, n_hidden))
        self.b1 = tf.Variable(tf.zeros([n_hidden], dtype=tf.float32))
        self.w2 = tf.Variable(tf.zeros([n_hidden, n_input], dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([n_input], dtype=tf.float32))

    def call(self, inputs):
        # 添加噪声
        noise = self.training_scale * tf.random.normal(tf.shape(inputs))
        noisy_inputs = inputs + noise

        # 编码层
        hidden = self.transfer(tf.matmul(noisy_inputs, self.w1) + self.b1)

        # 解码层
        reconstruction = tf.matmul(hidden, self.w2) + self.b2
        return reconstruction

    def compute_loss(self, inputs):
        reconstruction = self(inputs)
        return 0.5 * tf.reduce_sum(tf.pow(reconstruction - inputs, 2))

    def partial_fit(self, X):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(X)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss.numpy()

    def before_loss(self, X):
        return self.compute_loss(X).numpy()

    def transform(self, X):
        # 添加噪声
        noise = self.training_scale * tf.random.normal(tf.shape(X))
        noisy_X = X + noise

        # 获取隐藏层输出
        hidden = self.transfer(tf.matmul(noisy_X, self.w1) + self.b1)
        return hidden.numpy()

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=(1, self.n_hidden))
        return (tf.matmul(hidden, self.w2) + self.b2).numpy()

    def reconstruct(self, X):
        return self(X).numpy()

    def getWeights(self):
        return self.w1.numpy()

    def getBias(self):
        return self.b1.numpy()
