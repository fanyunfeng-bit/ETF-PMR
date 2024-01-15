import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.AVEDataset import AVEDataset
from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, CLClassifier, AClassifier, VClassifier
# from models.contrastive_model import AVConEncoder, AVConClassifier
from models.fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='CE', type=str, help='CE, CL, CEwCL, combine_modality')
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
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

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')
    parser.add_argument('--load_path', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')
    parser.add_argument('--load_path_other', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    return parser.parse_args()


def train_combine_classifier_epoch(args, epoch, audio_encoder, visual_encoder, classifier, device,
                                   dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    audio_encoder.eval()
    visual_encoder.eval()
    print("Start training classifier ... ")

    _loss = 0
    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        B = label.shape[0]

        optimizer.zero_grad()
        with torch.no_grad():
            a_features = audio_encoder(spec.unsqueeze(1).float())
            a_features = F.adaptive_avg_pool2d(a_features, 1)
            a_features = torch.flatten(a_features, 1)

            v_features = visual_encoder(image.float())
            (_, C, H, W) = v_features.size()
            v_features = v_features.view(B, -1, C, H, W)
            v_features = v_features.permute(0, 2, 1, 3, 4)
            v_features = F.adaptive_avg_pool3d(v_features, 1)
            v_features = torch.flatten(v_features, 1)

        _, _, out = classifier(a_features, v_features)

        loss = criterion(out, label)
        print('loss: ', loss)
        loss.backward()
        optimizer.step()

        _loss += loss.item()
    scheduler.step()
    return _loss / len(dataloader)


def valid_combine_classifier(args, audio_encoder, visual_encoder, classifier, device, dataloader):
    audio_encoder.eval()
    visual_encoder.eval()
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

            audio_features = audio_encoder(spec.unsqueeze(1).float())
            visual_features = visual_encoder(image.float())
            audio_features = F.adaptive_avg_pool2d(audio_features, 1)
            audio_features = torch.flatten(audio_features, 1)
            (_, C, H, W) = visual_features.size()
            visual_features = visual_features.view(B, -1, C, H, W)
            visual_features = visual_features.permute(0, 2, 1, 3, 4)
            visual_features = F.adaptive_avg_pool3d(visual_features, 1)
            visual_features = torch.flatten(visual_features, 1)

            _, _, out = classifier(audio_features, visual_features)

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
        audio_net = AClassifier(args)
        visual_net = VClassifier(args)
        model = AVClassifier(args)
    else:
        raise ValueError('Incorrect method!')

    audio_net.apply(weight_init)
    audio_net.to(device)
    visual_net.apply(weight_init)
    visual_net.to(device)
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

    # if args.dataset == 'VGGSound':
    #     n_classes = 309
    # elif args.dataset == 'KineticSound':
    #     n_classes = 31
    # elif args.dataset == 'CREMAD':
    #     n_classes = 6
    # elif args.dataset == 'AVE':
    #     n_classes = 28
    # else:
    #     raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:
        trainloss_file = args.logs_path + '/Method-' + 'Combine' + '/classifier_train_loss-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) + '-' + str(args.num_frame) + '.txt'
        if not os.path.exists(args.logs_path + '/Method-' + 'Combine'):
            os.makedirs(args.logs_path + '/Method-' + 'Combine')

        save_path = args.ckpt_path + '/Method-' + 'Combine' + '/classifier-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                    str(args.batch_size) + '-lr' + str(args.learning_rate) + '-' + str(args.num_frame)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        load_path1 = args.load_path   # AClassifier
        load_path2 = args.load_path_other  # VClassifier
        load_dict1 = torch.load(load_path1)
        load_dict2 = torch.load(load_path2)
        state_dict1 = load_dict1['model']
        state_dict2 = load_dict2['model']
        audio_net.load_state_dict(state_dict1)
        visual_net.load_state_dict(state_dict2)

        # model.audio_net.load_state_dict(audio_net.net.state_dict())
        # model.visual_net.load_state_dict(visual_net.net.state_dict())

        audio_encoder = audio_net.net
        visual_encoder = visual_net.net
        classifier = model.fusion_module

        optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9,
                              weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

        best_acc = 0
        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))

            batch_loss = train_combine_classifier_epoch(args, epoch, audio_encoder, visual_encoder, classifier, device,
                                                  train_dataloader, optimizer, scheduler)
            acc = valid_combine_classifier(args, audio_encoder, visual_encoder, classifier, device, test_dataloader)
            print('epoch: ', epoch, 'loss: ', batch_loss, 'acc: ', acc)

            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(acc) +
                              "\n")
            f_trainloss.flush()

            if acc > best_acc or (epoch + 1) % 10 == 0:
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
