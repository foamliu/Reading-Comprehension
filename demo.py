import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from config import batch_size
from data_gen import AiChallengerDataset
from data_gen import pad_collate

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    dset = AiChallengerDataset()
    dset.set_mode('test')
    test_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_acc = 0
    cnt = 0

    for batch_idx, data in enumerate(test_loader):
        contexts, questions, _, alternatives = data
        batch_size = contexts.size()[0]
        contexts = Variable(contexts.long().cuda())
        questions = Variable(questions.long().cuda())
        alternatives = Variable(alternatives.long().cuda())

        preds = model.forward(contexts, questions, alternatives)
        _, pred_ids = torch.max(preds, dim=1)
