import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from config import print_freq
from data_gen import AiChallengerDataset, pad_collate
from models import DMNPlus
from utils import parse_args, get_logger, AverageMeter


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)

    logger = get_logger()

    dset = AiChallengerDataset()
    vocab_size = len(dset.QA.VOCAB)

    model = DMNPlus(args.hidden_size, vocab_size, num_hop=3, qa=dset.QA)
    model.cuda()
    early_stopping_cnt = 0
    early_stopping_flag = False
    best_acc = 0
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(256):
        dset.set_mode('train')
        train_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

        model.train()

        losses = AverageMeter()
        accs = AverageMeter()

        if not early_stopping_flag:
            for i, data in enumerate(train_loader):
                optim.zero_grad()
                contexts, questions, answers, alternatives = data
                print('type(questions): ' + str(type(questions)))
                print('type(alternatives): ' + str(type(alternatives)))
                contexts = Variable(contexts.long().cuda())
                questions = Variable(questions.long().cuda())
                alternatives = Variable(alternatives.long().cuda())
                answers = Variable(answers.cuda())

                loss, acc = model.get_loss(contexts, questions, alternatives, answers)
                loss.backward()

                # Keep track of metrics
                losses.update(loss.item())
                accs.update(acc)

                if i % print_freq == 0:
                    logger.info(
                        '[Epoch {}][{}/{}] [Training]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          loss=losses,
                                                                          accs=accs
                                                                          ))
                optim.step()

            dset.set_mode('valid')
            valid_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

            model.eval()
            total_acc = 0
            cnt = 0
            for batch_idx, data in enumerate(valid_loader):
                contexts, questions, answers, alternatives = data
                batch_size = contexts.size()[0]
                contexts = Variable(contexts.long().cuda())
                questions = Variable(questions.long().cuda())
                alternatives = Variable(alternatives.long().cuda())
                answers = Variable(answers.cuda())

                _, acc = model.get_loss(contexts, questions, alternatives, answers)
                total_acc += acc * batch_size
                cnt += batch_size

            total_acc = total_acc / cnt
            if total_acc > best_acc:
                best_acc = total_acc
                best_state = model.state_dict()
                early_stopping_cnt = 0
            else:
                early_stopping_cnt += 1
                if early_stopping_cnt > 20:
                    early_stopping_flag = True

            logger.info('[Epoch {}] [Validate] Accuracy : {:.4f}'.format(epoch, total_acc))
            with open('log.txt', 'a') as fp:
                fp.write('[Epoch {}] [Validate] Accuracy : {:.4f}\n'.format(epoch, total_acc))
            if total_acc == 1.0:
                break
        else:
            logger.info('Early Stopping at Epoch {}, Valid Accuracy : {:.4f}'.format(epoch, best_acc))
            break

    dset.set_mode('test')
    test_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    test_acc = 0
    cnt = 0

    for batch_idx, data in enumerate(test_loader):
        contexts, questions, answers, alternatives = data
        batch_size = contexts.size()[0]
        contexts = Variable(contexts.long().cuda())
        questions = Variable(questions.long().cuda())
        alternatives = Variable(alternatives.long().cuda())
        answers = Variable(answers.cuda())

        model.load_state_dict(best_state)
        _, acc = model.get_loss(contexts, questions, alternatives, answers)
        test_acc += acc * batch_size
        cnt += batch_size
    logger.info('[Epoch {}] [Test] Accuracy : {:.4f}'.format(epoch, test_acc / cnt))
    os.makedirs('models', exist_ok=True)
    with open('models/epoch{}_acc{:.4f}.pth'.format(epoch, test_acc / cnt), 'wb') as fp:
        torch.save(model.state_dict(), fp)
    with open('log.txt', 'a') as fp:
        fp.write(
            '[Epoch {}] [Test] Accuracy : {:.4f}\n'.format(epoch, total_acc))


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
