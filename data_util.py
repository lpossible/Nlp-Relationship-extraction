"""Data util
@author:luo ping
@date:2020-8-19
"""
import nltk
import re


def clean_str(text):
    """clean str"""
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"<.*?>", "", text)
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
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
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


if __name__ == "__main__":
    # First step
    fw = open("data/step1.txt", 'w')
    with open("data/original.txt") as fr:
        lines = fr.readlines()
        for line in lines:
            line = clean_str(line)
            fw.write(line + "\n")
    fw.close()
    # Second step
    j = 0
    fw = open("data/train.txt", 'w')
    with open("data/step1.txt") as fr:
        for line in fr.readlines():
            if j % 4 == 0:
                line = re.sub(r"\n", " ", line)
                fw.write(line)
            if j % 2 == 1 and line != '\n':
                fw.write(line + '\n')
            j += 1
