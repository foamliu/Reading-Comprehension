import json
import pickle
import random

import jieba
from tqdm import tqdm

from config import train_path, valid_path, test_a_path, pickle_file


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def seg_line(line):
    return list(jieba.cut(line))


def get_raw_data():
    with open(train_path, 'r', encoding='utf-8') as file:
        train = file.readlines()
    with open(valid_path, 'r', encoding='utf-8') as file:
        valid = file.readlines()
    with open(test_a_path, 'r', encoding='utf-8') as file:
        test = file.readlines()
    return train, valid, test


def get_unindexed_qa(lines):
    data = []

    for line in lines:
        item = json.loads(line)
        question = item['query']
        doc = item['passage']
        alternatives = item['alternatives']
        if 'answer' in item.keys():
            answer = item['answer']
        else:
            answer = '<PAD>'
        data.append({'Q': question, 'C': doc.split('。'), 'A': answer, 'AL': alternatives.split('|')})
    return data


def get_indexed_qa(raw_data):
    print('get indexed qa...')
    unindexed = get_unindexed_qa(raw_data)
    questions = []
    contexts = []
    answers = []
    alternatives = []
    for qa in tqdm(unindexed):
        context = [seg_line(c.strip()) + ['<EOS>'] for c in qa['C']]

        for con in context:
            for token in con:
                build_vocab(token)
        context = [[QA.VOCAB[token] for token in sentence] for sentence in context]
        question = seg_line(qa['Q']) + ['<EOS>']

        for token in question:
            build_vocab(token)
        question = [QA.VOCAB[token] for token in question]

        alternative = qa['AL']
        for token in alternative:
            build_vocab(token)
        alternative = [QA.VOCAB[token] for token in alternative]
        random.shuffle(alternative)

        answer = qa['A']
        if not answer == '<PAD>':
            build_vocab(answer)
            answer = QA.VOCAB[answer]
            answer = alternative.index(answer)
        else:
            answer = 0

        contexts.append(context)
        questions.append(question)
        answers.append(answer)
        alternatives.append(alternative)
    return contexts, questions, answers, alternatives


def build_vocab(token):
    if not token in QA.VOCAB:
        next_index = len(QA.VOCAB)
        QA.VOCAB[token] = next_index
        QA.IVOCAB[next_index] = token


if __name__ == '__main__':
    raw_train, raw_valid, raw_test = get_raw_data()
    QA = adict()
    QA.VOCAB = {'<PAD>': 0, '<EOS>': 1}
    QA.IVOCAB = {0: '<PAD>', 1: '<EOS>'}
    data = dict()
    data['VOCAB'] = QA.VOCAB
    data['IVOCAB'] = QA.IVOCAB
    data['train'] = get_indexed_qa(raw_train)
    data['valid'] = get_indexed_qa(raw_valid)
    data['test'] = get_indexed_qa(raw_test)
    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)
