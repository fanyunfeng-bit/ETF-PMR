import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.AVEDataset import AVEDataset
from dataset.CGMNIST import CGMNISTDataset
from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, AClassifier, VClassifier, GrayClassifier, ColoredClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', default='audio', type=str, help='audio, visual')
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    return parser.parse_args()


def train_epoch(args, epoch, model, device,
                dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0
    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        B = label.shape[0]
        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.dataset == 'CGMNIST':
            if args.modality == 'gray':
                out = model(spec)
            elif args.modality == 'colored':
                out = model(image)
            else:
                raise ValueError('error.')
        else:
            if args.modality == 'audio':
                out = model(spec.unsqueeze(1).float())
            elif args.modality == 'visual':
                out = model(image.float(), B)
            else:
                raise ValueError('error.')

        loss = criterion(out, label)
        # print('loss: ', loss)
        loss.backward()
        optimizer.step()
        _loss += loss.item()
    scheduler.step()
    return _loss / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            B = label.shape[0]

            if args.dataset == 'CGMNIST':
                if args.modality == 'gray':
                    out = model(spec)
                elif args.modality == 'colored':
                    out = model(image)
                else:
                    raise ValueError('error.')
            else:
                if args.modality == 'audio':
                    out = model(spec.unsqueeze(1).float())
                elif args.modality == 'visual':
                    out = model(image.float(), B)
                else:
                    raise ValueError('error.')

            prediction = softmax(out)
            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0  # what is label[i]
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
    return sum(acc) / sum(num)


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:'+str(args.gpu) if args.use_cuda else 'cpu')
    
    if args.dataset == 'CGMNIST':
        if args.modality == 'gray':
            model = GrayClassifier(args)
        elif args.modality == 'colored':
            model = ColoredClassifier(args)
        else:
            raise ValueError('In correct modality choice.')
    else:
        if args.modality == 'audio':
            model = AClassifier(args)
        elif args.modality == 'visual':
            model = VClassifier(args)
        else:
            raise ValueError('In correct modality choice.')

    model.apply(weight_init)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    elif args.dataset == 'CGMNIST':
        train_dataset = CGMNISTDataset(args, mode='train')
        test_dataset = CGMNISTDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:

        trainloss_file = args.logs_path + '/unimodality-' + args.modality + '/train_loss-' + args.dataset + '-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) + '.txt'
        if not os.path.exists(args.logs_path + '/unimodality-' + args.modality):
            os.makedirs(args.logs_path + '/unimodality-' + args.modality)

        save_path = args.ckpt_path + '/unimodality-' + args.modality + '/model-' + args.dataset + '-bsz' + \
                    str(args.batch_size) + '-lr' + str(args.learning_rate)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        best_acc = 0.0
        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))

            batch_loss = train_epoch(args, epoch, model, device,
                                     train_dataloader, optimizer, scheduler)
            acc = valid(args, model, device, test_dataloader)
            print('epoch: ', epoch, 'acc: ', acc, 'loss: ', batch_loss)

            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(acc) +
                              "\n")
            f_trainloss.flush()

            if acc > best_acc or (epoch + 1) % 10 == 0:
                if acc > best_acc:
                    best_acc = float(acc)

                # save model parameter
                print('Saving model....')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },
                    os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
                )
                print('Saved model!!!')
        f_trainloss.close()


if __name__ == '__main__':
    main()