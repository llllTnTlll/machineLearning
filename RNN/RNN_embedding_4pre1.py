import tensorflow as tf
import os
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

word = 'abcdefghijklmnopqrstuvwxyz'
word_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12,
              'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 172, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
              'y': 24, 'z': 25}
training_set_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
x_train = []
y_train = []

for i in range(4, 26):
    x_train.append(training_set_scale[i-4:i])
    y_train.append(training_set_scale[i])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)

# 在不使用embedding时SimpleRNN要求输入符合  [输入样本数，循环核时间展开步数，每个时间步输入特征数]
# 使用embedding后embedding层要求输入符合    [输入样本数，循环核时间展开步数]
x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.models.Sequential([
    Embedding(26, 2),
    SimpleRNN(10),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './Checkpoint/RNN_embedding_4pre1.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')    # 无测试集，因此根据loss保存最优

history = model.fit(x_train, y_train, batch_size=32, epochs=200, callbacks=[cp_callback])

model.summary()

pre_word = input('please enter pre word')
alphabet = [word_to_id[a] for a in pre_word]
alphabet = np.array(alphabet)
alphabet = np.reshape(alphabet, (1, 4))
result = model.predict(alphabet)
pred = int(np.argmax(result, axis=1))
print('pred word: ' + word[pred])
