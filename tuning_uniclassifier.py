import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.CramedDataset import CramedDataset
from dataset.AVEDataset import AVEDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, CLClassifier, AClassifier, VClassifier
# from models.contrastive_model import AVConEncoder, AVConClassifier
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='CE', type=str, help='CE, CL, CEwCL, combine_modality')
    parser.add_argument('--modality_type', default='uni', type=str, help='uni, multi')
    parser.add_argument('--modulation', default='Normal', type=str, help='Normal, OGM_GE, Acc, Proto')
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--train_modality', default='audio', type=str,
                        help='Fine tuning the corresponding classifier')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--trained_epoch', default=99, type=int)

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')
    parser.add_argument('--load_path', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    return parser.parse_args()


def train_uniclassifier_epoch(args, epoch, encoder, classifier, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    encoder.eval()
    print("Start training classifier ... ")

    _loss = 0
    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        B = label.shape[0]

        optimizer.zero_grad()
        if args.train_modality == 'audio':
            with torch.no_grad():
                features = encoder(spec.unsqueeze(1).float())
                features = F.adaptive_avg_pool2d(features, 1)
                features = torch.flatten(features, 1)

            if args.fusion_method == 'sum':
                pass
            elif args.fusion_method == 'concat':
                om_features = torch.zeros_like(features).to(device)  # om: other modality
                features = torch.cat((features, om_features), dim=1)
            elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                pass
            out = classifier(features)
        elif args.train_modality == 'visual':
            with torch.no_grad():
                features = encoder(image.float())
                (_, C, H, W) = features.size()
                features = features.view(B, -1, C, H, W)
                features = features.permute(0, 2, 1, 3, 4)
                features = F.adaptive_avg_pool3d(features, 1)
                features = torch.flatten(features, 1)

            if args.fusion_method == 'sum':
                pass
            elif args.fusion_method == 'concat':
                om_features = torch.zeros_like(features).to(device)  # om: other modality
                features = torch.cat((om_features, features), dim=1)
            elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                pass
            out = classifier(features)
        else:
            raise ValueError('Incorrect modality')

        loss = criterion(out, label)
        print('loss: ', loss)
        loss.backward()
        optimizer.step()

        _loss += loss.item()
    scheduler.step()
    return _loss / len(dataloader)


def valid_uniclassifier(args, encoder, classifier, device, dataloader):
    encoder.eval()
    classifier.eval()

    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            B = label.shape[0]

            if args.train_modality == 'audio':
                features = encoder(spec.unsqueeze(1).float())
                features = F.adaptive_avg_pool2d(features, 1)
                features = torch.flatten(features, 1)
                if args.fusion_method == 'sum':
                    pass
                elif args.fusion_method == 'concat':
                    om_features = torch.zeros_like(features).to(device)  # om: other modality
                    features = torch.cat((features, om_features), dim=1)
                elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                    pass
                out = classifier(features)
            elif args.train_modality == 'visual':
                features = encoder(image.float())
                (_, C, H, W) = features.size()
                features = features.view(B, -1, C, H, W)
                features = features.permute(0, 2, 1, 3, 4)
                features = F.adaptive_avg_pool3d(features, 1)
                features = torch.flatten(features, 1)
                if args.fusion_method == 'sum':
                    pass
                elif args.fusion_method == 'concat':
                    om_features = torch.zeros_like(features).to(device)  # om: other modality
                    features = torch.cat((om_features, features), dim=1)
                elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                    pass
                out = classifier(features)
            else:
                raise ValueError('Incorrect modality')

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

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.method == 'CE' or args.method == 'CE_Proto':
        model = AVClassifier(args)
    else:
        raise ValueError('Incorrect method!')

    model.apply(weight_init)
    model.to(device)

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
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:
        if args.modality_type == 'uni':
            trainloss_file = args.logs_path + '/Method-' + args.method + '/' + args.modality_type + '-' + args.train_modality + '/classifier_train_loss-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                             str(args.batch_size) + '-lr' + str(args.learning_rate) + \
                             '-modulation' + str(args.modulation) + str(args.alpha) + \
                             '-trained_epoch' + str(args.trained_epoch) + '.txt'
            if not os.path.exists(args.logs_path + '/Method-' + args.method + '/' + args.modality_type + '-' + args.train_modality):
                os.makedirs(args.logs_path + '/Method-' + args.method + '/' + args.modality_type + '-' + args.train_modality)

            save_path = args.ckpt_path + '/Method-' + args.method + '/' + args.modality_type + '-' + args.train_modality + '/classifier-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                        str(args.batch_size) + '-lr' + str(args.learning_rate) + \
                             '-modulation' + str(args.modulation) + str(args.alpha) + \
                             '-trained_epoch' + str(args.trained_epoch)
        else:
            raise ValueError('error')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        # load trained encoder
        if args.method == 'CE' or args.method == 'CE_Proto':
            if args.modality_type == 'uni':
                load_path = args.load_path
                load_dict = torch.load(load_path)
                state_dict = load_dict['model']
                model.load_state_dict(state_dict)
                if args.train_modality == 'audio':
                    encoder = model.audio_net
                    if args.fusion_method == 'sum':
                        classifier = model.fusion_module.fc_x    # 512
                    elif args.fusion_method == 'concat':
                        classifier = model.fusion_module.fc_out  # 1024
                    elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                        classifier = model.fusion_module.fc_out  # 512
                elif args.train_modality == 'visual':
                    encoder = model.visual_net
                    if args.fusion_method == 'sum':
                        classifier = model.fusion_module.fc_y  # 512
                    elif args.fusion_method == 'concat':
                        classifier = model.fusion_module.fc_out  # 1024
                    elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                        classifier = model.fusion_module.fc_out  # 512
                else:
                    raise ValueError('Incorrect modality to be trained')

                optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9,
                                      weight_decay=1e-4)
                scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

                best_acc = 0
                for epoch in range(args.epochs):
                    print('Epoch: {}: '.format(epoch))

                    batch_loss = train_uniclassifier_epoch(args, epoch, encoder, classifier, device,
                                                           train_dataloader, optimizer, scheduler)
                    acc = valid_uniclassifier(args, encoder, classifier, device, test_dataloader)
                    print('epoch: ', epoch, 'loss: ', batch_loss, 'acc: ', acc)

                    f_trainloss.write(str(epoch) +
                                      "\t" + str(batch_loss) +
                                      "\t" + str(acc) +
                                      "\n")
                    f_trainloss.flush()

                    if acc > best_acc or (epoch + 1) % args.epochs == 0:
                        if acc > best_acc:
                            best_acc = float(acc)
                        print('Saving model....')
                        torch.save(
                            {
                                'classifier': classifier.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()
                            },
                            os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
                        )
                        print('Saved model!!!')

        f_trainloss.close()


if __name__ == '__main__':
    main()
