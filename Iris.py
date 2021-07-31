import matplotlib.pyplot as plt
from sklearn import datasets
from pandas import DataFrame
import pandas as pd
import numpy as np
import tensorflow as tf

# 数据处理

# 从Iris数据集中读取数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 加入表头
x_data_withlable = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# 设置表头对齐
pd.set_option('display.unicode.east_asian_width', True)
# 添加结果列
x_data_withlable['类别'] = y_data
print('x_data sadd index: \n', x_data_withlable)

# 使用统一的随机数种子打乱数据集顺序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

# 截取前120行做训练集
# 后30行做测试集
x_train = x_data[:-30]
x_test = x_data[-30:]
y_train = y_data[:-30]
y_test = y_data[-30:]
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 将训练集和测试集相关联
# 以32个为一组打包
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 模型训练

# 定义关键参数
lr = 0.2
loss_history = []
acc_history = []
epoch = 500

# 初始化神经网络的权重和偏移量
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))

# 网络训练
loss_all = 0
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_true = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_true - y))
            loss_all += loss
        grad = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grad[0])
        b1.assign_sub(lr * grad[1])
    print('Epoch {}, loss {}'.format(epoch, loss_all / 4))
    loss_history.append(loss_all / 4)
    loss_all = 0

    # 正确率检测
    correct_num = 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        correct_num += correct
        total_num = x_test.shape[0]
    acc = correct_num / total_num
    acc_history.append(acc)
    print('Test acc', acc)

# 绘制损失变化曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss_history, label='$Loss$')
plt.legend()
plt.show()

# 绘制正确率变化曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(acc_history, label='$Accuracy$')
plt.legend()
plt.show()