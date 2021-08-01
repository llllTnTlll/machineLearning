import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

(x_test, y_test), (x_train, y_train) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train/255, x_test/255


class fasion_model(Model):
    def __init__(self):
        super(fasion_model, self).__init__()
        self.flatten = Flatten()
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.d1(x)
        x = self.d2(x)
        y = self.d3(x)
        return y


model = fasion_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='sparse_categorical_accuracy')
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)

model.summary()