"""
@author:luo ping
@date:2020-9-1
"""
import re
import pickle


# clean the data
def clean_str(text):
    """clean str"""
    text = re.sub(r"\n", "", text)
    text = re.sub(r"(.*?)\t", "", text)
    text = re.sub(r"\"", "", text)
    # 替换e1
    text = re.sub(r"<e1.*?e1>", "_e1_", text)
    # 替换e2
    text = re.sub(r"<e2.*?e2>", "_e2_", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# 数据处理
# j = 0
# fw = open("data/position.txt", "w")
# with open("data/position_original.txt") as fr:
#     lines = fr.readlines()
#     for line in lines:
#         if line != '\n':
#             if j % 4 == 0:
#                 line = clean_str(line)
#                 fw.write(line + '\n')
#                 fw.write("\n")
#         j += 1
# 提取位置信息
data_pos1, data_pos2 = [], []
with open("data/position.txt") as fr:
    lines = fr.readlines()
    for line in lines:
        if line != '\n':
            pos1, pos2 = [], []
            e1_pos = line.split().index("_e1_")
            e2_pos = line.split().index("_e2_")
            for i in range(len(line.split())):
                pos1.append(i - e1_pos + len(line.split()) - 1)
                pos2.append(i - e2_pos + len(line.split()) - 1)
            data_pos1.append(pos1)
            data_pos2.append(pos2)
with open("data/pos1_info.pkl", "wb") as fw:
    pickle.dump(data_pos1, fw)
with open("data/pos2_info.pkl", "wb") as fw:
    pickle.dump(data_pos2, fw)
