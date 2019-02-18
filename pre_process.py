import json
import os
import pickle

import jieba
from tqdm import tqdm

from config import train_folder, train_filename, min_word_freq


def seg_line(line):
    return list(jieba.cut(line))


def seg_data(path):
    print('start process ', path)
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []

    for line in tqdm(lines):
        item = json.loads(line)
        question = item['query']
        doc = item['passage']
        alternatives = item['alternatives']
        data.append([seg_line(question), seg_line(doc), alternatives.split('|'), item['query_id']])
    return data


def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:3]]
    print('vocab size ', len(wordCount))
    return wordCount


def build_vocab(wordCount, threshold):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in wordCount:
        if wordCount[word] >= threshold:
            if word not in vocab:
                vocab[word] = len(vocab)
        else:
            chars = list(word)
            for char in chars:
                if char not in vocab:
                    vocab[char] = len(vocab)
    print('processed vocab size ', len(vocab))
    return vocab


def main():
    # 对原始语料分词  注意答案的分词方式是通过语料中的 | 直接分词的
    path = os.path.join(train_folder, train_filename)
    data = seg_data(path)

    # [[question,doc,answer,id],[],[]...]  word_count = {"":}
    word_count = build_word_count(data)

    with open('data/word-count.pickle', 'wb') as f:
        pickle.dump(word_count, f)

    vocab = build_vocab(word_count, min_word_freq)  # 对分词之后的建立所有，小于threshold的词将被分成char，然后加入词表
    with open('data/vocab.pickle', 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == '__main__':
    main()
