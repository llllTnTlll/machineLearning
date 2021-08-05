import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
np.set_printoptions(threshold=np.inf)    # 设置np展示全部不省略

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255


class MNIST_model(Model):
    def __init__(self):
        super(MNIST_model, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.d1(x)
        y = self.d2(x)
        return y


model = MNIST_model()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='sparse_categorical_accuracy')

checkpoint_save_path = "./mnist.ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
    print('----load model----')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

# 展示训练参数
print(model.trainable_variables)
# 写入指定为文件
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 展示正确率与损失函数变化
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()