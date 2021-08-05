import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense
import os
from matplotlib import pyplot as plt

class ConvBNRelu(Model):
    def __init__(self, ch, kernalsz, stride, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.Sequential([
            Conv2D(ch, kernel_size=kernalsz, strides=stride, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs)
        return x

class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernalsz=1, stride=strides)
        self.c2_1 = ConvBNRelu(ch, kernalsz=1, stride=strides)
        self.c2_2 = ConvBNRelu(ch, kernalsz=3, stride=1)
        self.c3_1 = ConvBNRelu(ch, kernalsz=1, stride=strides)
        self.c3_2 = ConvBNRelu(ch, kernalsz=5, stride=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernalsz=1, stride=strides)

    def call(self, inputs, training=None, mask=None):
        x1 = self.c1(inputs)
        x2_1 = self.c2_1(inputs)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(inputs)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(inputs)
        x4_2 = self.c4_2(x4_1)
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_chanels = init_ch
        self.out_chanels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch, kernalsz=3, stride=1)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_chanels, strides=2)
                else:
                    block = InceptionBlk(self.out_chanels, strides=1)
                self.blocks.add(block)
            self.out_chanels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = Inception10(num_blocks=2, num_classes=10)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_path = './CIFAR10_InceptionNET/cifar10.ckpt'
if os.path.exists(checkpoint_path + '.index'):
    model.load_weights(filepath=checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True)

history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=cp_callback)

model.summary()
file = open('./CIFAR10_InceptionNET/weight.txt', 'w')
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
