import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
import au_class as au
import pandas as pd

def standard_scale(X_train):# 数据标准化函数，作用：对输入数据进行标准化（零均值、单位方差）。输出：标准化后的数据。
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def get_random_block_from_data(data, batch_size):# 随机块采样函数，作用：从数据中随机选取一个大小为 batch_size 的块，用于小批量训练。注意：若数据量不足 batch_size，可能导致索引错误。
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

"""
参数说明：
x_train：输入训练数据。
input_size：输入特征维度。
training_epochs：每个自编码器的训练轮数。
batch_size：小批量大小。
display_step：每多少轮输出一次训练信息。
lowsize：隐藏层的节点数（固定为所有层的隐藏层大小）。
hidden_size：隐藏层的数量（如 hidden_size=[256, 128] 表示堆叠两层自编码器）。
初始化：创建 len(hidden_size) 个自编码器，每个的隐藏层大小均为 lowsize。
"""
#[708,160],160,20,16,1,100,[100]
def DAE(x_train,input_size,training_epochs,batch_size,display_step,lowsize,hidden_size):
    sdne = []
    ###initialize
    for i in range(len(hidden_size)):
        ae = au.Autoencoder(
            n_input=input_size,
            n_hidden=lowsize,
            transfer_function=tf.nn.softplus,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            scale=0.2)
        sdne.append(ae)

    Hidden_feature = []
    for j in range(len(hidden_size)):
        if j == 0:
            X_train = standard_scale(x_train)
        else:
            X_train_pre = X_train
            X_train = sdne[j - 1].transform(X_train_pre)
            Hidden_feature.append(X_train)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)

            for batch in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)

                cost = sdne[j].partial_fit(batch_xs)

                avg_cost += cost / X_train.shape[0] * batch_size
            if epoch % display_step == 0:
                print("Epoch:", "%4d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))

        if j == 0:
            feat0 = sdne[0].transform(standard_scale(x_train))
            data1 = pd.DataFrame(feat0)
            print(data1.shape)
            np.set_printoptions(suppress=True)
    return data1

