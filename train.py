"""
@author:luo ping
@date:2020-9-2
"""
import tensorflow as tf
import os
from keras.preprocessing import sequence
from keras.utils import to_categorical
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Conv1D, Dense, Dropout, Flatten
from Rlayers import SEmbedding

# 标签字典
class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}

tf.set_random_seed(4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 测试数据路径
test_data_path = './data/test.txt'
# 训练数据路径
train_data_path = './data/train.txt'
# 词典数据路径
vocab_path = './data/vocab.txt'
# 词向量数据路径
word2vec_data_path = './data/wordvec.txt'
# 词向量pkl路径
word_vec_pkl = './data/word_vec.pkl'


# 开始数据处理
def get_data(data_path, word_dict=None, label_dict=None, mode=None):
    # 处理词向量数据
    if mode == 'vec':
        words_vec = []
        with open(data_path, encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line != '\n':
                    word_vec = line.strip().split()[1:]
                    words_vec.append(word_vec)
        all_vec = list()
        for each in words_vec:
            each_vec = []
            for char in each:
                each_vec.append(eval(char))
            all_vec.append(each_vec)
        all_vec = np.asarray(all_vec)
        with open(word_vec_pkl, 'wb') as fw:
            pickle.dump(all_vec, fw)
        return True
    # 建立词典
    elif mode == 'vocab':
        word_list = list()
        with open(data_path, encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line != '\n':
                    word_list.append(line.strip())
        # 特殊词
        special_word = ['pad', 'unknown']
        # 对词典表进行汇总
        word_list = special_word + word_list
        word_dict = dict()
        for key, value in enumerate(word_list):
            word_dict[value] = key
        return word_dict
    # 处理训练和测试数据
    else:
        data, labels = [], []
        with open(data_path, encoding='utf-8') as fr:
            lines = fr.readlines()
        for line in lines:
            if line != '\n':
                sentence = line.strip().split()[:-1]
                label = line.strip().split()[-1]
                sentence2data = [word_dict[each] if each in word_dict else word_dict['unknown'] for each in sentence]
                data.append(sentence2data)
                labels.append(label_dict[label])
        return data, labels


# 处理词向量数据
vec_data_processing = get_data(word2vec_data_path, mode='vec')
# 建立中文字词典
words_dict = get_data(vocab_path, mode='vocab')
# 得到训练数据和测试数据
train_data, train_labels = get_data(train_data_path, word_dict=words_dict, label_dict=class2label)
test_data, test_labels = get_data(test_data_path, word_dict=words_dict, label_dict=class2label)
# padding 数据
train_data = sequence.pad_sequences(train_data, maxlen=100, padding='post')
# train_labels = sequence.pad_sequences(train_labels, maxlen=100, padding='post')
test_data = sequence.pad_sequences(test_data, maxlen=100, padding='post')
# test_labels = sequence.pad_sequences(test_labels, maxlen=100, padding='post')
# 将标签转换为one-hot编码
train_labels = to_categorical(train_labels, len(class2label))
test_labels = to_categorical(test_labels, len(class2label))
# 位置数据
with open("data/pos1_info.pkl", "rb") as fr:
    pos1_data = pickle.load(fr)
with open("data/pos2_info.pkl", "rb") as fr:
    pos2_data = pickle.load(fr)
pos1_data = sequence.pad_sequences(pos1_data, maxlen=100, padding='post', value=199)
pos2_data = sequence.pad_sequences(pos2_data, maxlen=100, padding='post', value=199)
# 建立CNN模型
inputs = Input(shape=(100,), dtype='int32', name='inputs')
pos1 = Input(shape=(100,), dtype='int32', name='pos1')
pos2 = Input(shape=(100,), dtype='int32', name='pos2')
x = SEmbedding()([inputs, pos1, pos2])
x = Dropout(0.5)(x)
x = Conv1D(100, kernel_size=3, padding='same', activation='relu')(x)
x = Conv1D(1, kernel_size=3, padding='same', activation='relu')(x)
x = Flatten()(x)
outputs = Dense(len(class2label), activation=tf.keras.activations.softmax)(x)
model = Model(inputs=[inputs, pos1, pos2], outputs=outputs)
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['acc'])
model.summary()
model.fit(x=[train_data, pos1_data, pos2_data], y=train_labels, epochs=100, validation_split=0.2)
model.save("model/test.h5")
