from sklearn import datasets
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

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
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

model.summary()