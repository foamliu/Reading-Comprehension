import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from config import train_path, valid_path, test_a_path


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


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
        self.vocab_path = 'data/vocab.pkl'
        self.mode = mode
        raw_data, raw_valid, raw_test = get_raw_data(mode)
        self.QA = adict()
        self.QA.VOCAB = {'<PAD>': 0, '<EOS>': 1}
        self.QA.IVOCAB = {0: '<PAD>', 1: '<EOS>'}
        self.data = self.get_indexed_qa(raw_data)
        self.valid = self.get_indexed_qa(raw_valid)
        self.test = self.get_indexed_qa(raw_test)

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
        questions = []
        contexts = []
        answers = []
        for qa in raw_data:
            context = [c.lower().split() + ['<EOS>'] for c in qa['C']]

            for con in context:
                for token in con:
                    self.build_vocab(token)
            context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
            question = qa['Q'].lower().split() + ['<EOS>']

            for token in question:
                self.build_vocab(token)
            question = [self.QA.VOCAB[token] for token in question]

            self.build_vocab(qa['A'].lower())
            answer = self.QA.VOCAB[qa['A'].lower()]

            contexts.append(context)
            questions.append(question)
            answers.append(answer)
        return (contexts, questions, answers)

    def build_vocab(self, token):
        if not token in self.QA.VOCAB:
            next_index = len(self.QA.VOCAB)
            self.QA.VOCAB[token] = next_index
            self.QA.IVOCAB[next_index] = token


def get_raw_data():
    with open(train_path, 'r') as file:
        train = file.readlines()
    with open(valid_path, 'r') as file:
        valid = file.readlines()
    with open(test_a_path, 'r') as file:
        test = file.readlines()
    return train, valid, test


if __name__ == '__main__':
    dset_train = AiChallengerDataset()
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers = data
        break
