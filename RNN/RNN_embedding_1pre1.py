import tensorflow as tf
import os
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

word = 'abcde'
word_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
x_train = [word_to_id['a'], word_to_id['b'], word_to_id['c'], word_to_id['d'], word_to_id['e']]
y_train = [word_to_id['b'], word_to_id['c'], word_to_id['d'], word_to_id['e'], word_to_id['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)

# 在不使用embedding时SimpleRNN要求输入符合  [输入样本数，循环核时间展开步数，每个时间步输入特征数]
# 使用embedding后embedding层要求输入符合    [输入样本数，循环核时间展开步数]
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)

model = tf.keras.models.Sequential([
    Embedding(5, 2),
    SimpleRNN(3),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './Checkpoint/RNN_embedding_1pre1.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')    # 无测试集，因此根据loss保存最优

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])

model.summary()

pre_word = input('please enter pre word')
alphabet = word_to_id[pre_word]
alphabet = np.array(alphabet)
alphabet = np.reshape(alphabet, (1, 1))
result = model.predict(alphabet)
pred = int(np.argmax(result, axis=1))
print('pred word: ' + word[pred])

