import json
import pickle

import jieba
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
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
            answer = 'NA'
        data.append({'Q': question, 'C': doc.split('。'), 'A': answer, 'alter': alternatives.split('|')})
    return data


def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer)
    return default_collate(batch)


class AiChallengerDataset(Dataset):
    def __init__(self, mode='train'):
        # self.vocab_path = 'data/vocab.pkl'
        self.mode = mode
        # raw_train, raw_valid, raw_test = get_raw_data()
        # self.QA = adict()
        # self.QA.VOCAB = {'<PAD>': 0, '<EOS>': 1}
        # self.QA.IVOCAB = {0: '<PAD>', 1: '<EOS>'}
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.train = data['train']
        self.valid = data['valid']
        self.test = data['test']

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'valid':
            return len(self.valid[0])
        elif self.mode == 'test':
            return len(self.test[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers = self.train
        elif self.mode == 'valid':
            contexts, questions, answers = self.valid
        elif self.mode == 'test':
            contexts, questions, answers = self.test
        return contexts[index], questions[index], answers[index]

    def get_indexed_qa(self, raw_data):
        print('get indexed qa...')
        unindexed = get_unindexed_qa(raw_data)
        questions = []
        contexts = []
        answers = []
        for qa in tqdm(unindexed):
            context = [seg_line(c.strip()) + ['<EOS>'] for c in qa['C']]

            for con in context:
                for token in con:
                    self.build_vocab(token)
            context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
            question = seg_line(qa['Q']) + ['<EOS>']

            for token in question:
                self.build_vocab(token)
            question = [self.QA.VOCAB[token] for token in question]

            self.build_vocab(qa['A'])
            answer = self.QA.VOCAB[qa['A']]

            contexts.append(context)
            questions.append(question)
            answers.append(answer)
        return contexts, questions, answers

    def build_vocab(self, token):
        if not token in self.QA.VOCAB:
            next_index = len(self.QA.VOCAB)
            self.QA.VOCAB[token] = next_index
            self.QA.IVOCAB[next_index] = token


if __name__ == '__main__':
    dset_train = AiChallengerDataset()
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers = data
        break
    print(len(dset_train.QA.VOCAB))
