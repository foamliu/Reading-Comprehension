import pickle

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from config import pickle_file


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer, alternative = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        alternative = np.array(alternative)
        batch[i] = (context, question, answer, alternative)
        print('context.shape: ' + str(context.shape))
        print('question.shape: ' + str(question.shape))
        print('alternative.shape: ' + str(alternative.shape))
        print('answer: ' + str(answer))
    return default_collate(batch)


class AiChallengerDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.QA = adict()
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.QA.VOCAB = data['VOCAB']
        self.QA.IVOCAB = data['IVOCAB']
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
            contexts, questions, answers, alternatives = self.train
        elif self.mode == 'valid':
            contexts, questions, answers, alternatives = self.valid
        elif self.mode == 'test':
            contexts, questions, answers, alternatives = self.test
        return contexts[index], questions[index], answers[index], alternatives[index]


if __name__ == '__main__':
    dset_train = AiChallengerDataset()
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers, alternatives = data
        print('type(contexts): ' + str(type(contexts)))
        print('type(questions): ' + str(type(questions)))
        print('type(alternatives): ' + str(type(alternatives)))
        print('type(answers): ' + str(type(answers)))
        break
    print(len(dset_train.QA.VOCAB))
