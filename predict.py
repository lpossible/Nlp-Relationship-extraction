"""
@author:Luo Ping
@date:2020-09-07-10:53
"""
from keras.models import load_model
import numpy as np
from keras.preprocessing import sequence
import os
from Rlayers import SEmbedding, PositionEmbedding
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
model = load_model("./model/test.h5",
                   custom_objects={"SEmbedding": SEmbedding, "PositionEmbedding": PositionEmbedding})
special_words = ['pad', 'unknown']  # 特殊词表示
# 读取字符词典文件
with open('./data/vocab.txt', encoding="utf8") as fr:
    char_vocabs = [line.strip() for line in fr.readlines()]
char_vocabs = special_words + char_vocabs

# 字符和索引编号对应,enumerate从0开始
id_to_vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab_to_id = {char: idx for idx, char in id_to_vocab.items()}
# "BIO"标记的标签
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
# 索引和BIO标签对应
idx2label = {idx: label for label, idx in class2label.items()}


def read_corpus(corpus_path, vocab2idx):
    """read_corpus"""
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        if line != '\n':
            sentence = line.strip().split()[:-1]
            sentence2data = [vocab2idx[each] if each in vocab2idx else vocab2idx['unknown'] for each in sentence]
            data.append(sentence2data)
    return data


# 提取所有句子标签
test_data_labels = []
with open("data/test.txt", encoding="utf-8") as fr:
    for line in fr.readlines():
        if line != '\n':
            test_data_labels.append(class2label[line.strip().split()[-1]])
# 数据路径
data_path = './data/test.txt'
# 数据转为id序列,并保存数据序列
test_data = read_corpus(data_path, vocab_to_id)

# padding数据
MAX_LEN = 100
test_data = sequence.pad_sequences(test_data, MAX_LEN, padding='post')
output = model.predict([test_data])
# 提取标签
output_labels = []
for each in output:
    output_labels.append(np.argmax(each))
# 写入预测文件
with open("eval/prediction.txt", "w", encoding="utf-8") as fw:
    for i in range(len(output_labels)):
        label = label2class[output_labels[i]]
        fw.write("{}\t{}\n".format(i, label))
# 模型评估
process = subprocess.Popen(
    ["perl", "eval/semeval2010_task8_scorer-v1.2.pl", "eval/prediction.txt", "eval/test_truth.txt"],
    stdout=subprocess.PIPE, shell=True)
while process.poll() is None:
    for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
        print(line)
