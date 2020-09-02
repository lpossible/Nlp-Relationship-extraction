"""
@author:Luo Ping
@date:2020-09-02-9:26
"""
from gensim.models import Word2Vec

# 语料路径
train_data_path = 'data/train.txt'
test_data_path = 'data/test.txt'
# 数据序列
data = []
# 对于已分词且标注的语料，需要形成句子序列
# 提取训练语料数据
with open(train_data_path, encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if line != '\n':
            data.append(line.strip().split()[:-1])
# 提取测试语料数据
with open(test_data_path, encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if line != '\n':
            data.append(line.strip().split()[:-1])
model = Word2Vec(data, min_count=2)
model.train(data, total_examples=len(data), epochs=32)
model.wv.save_word2vec_format(fname='./data/wordvec.txt', binary=False)
model.save('./model/wordvec.w2v')
