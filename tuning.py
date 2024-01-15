import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, CLClassifier, AClassifier, VClassifier
# from models.contrastive_model import AVConEncoder, AVConClassifier
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F


def get_arguments():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--load_path_a', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')
    parser.add_argument('--load_path_v', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    return parser.parse_args()


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def train_uniclassifier_epoch(args, epoch, audio_net, visual_net, device, dataloader, optimizer_a, optimizer_v,
                              scheduler):
    criterion = nn.CrossEntropyLoss()

    audio_net.train()
    visual_net.train()
    # encoder.eval()
    print("Start training classifier ... ")

    _loss = 0
    _a_angle = 0
    _v_angle = 0

    angle_file = args.logs_path + '/combine' + '/angle-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                 str(args.batch_size) + '-lr' + str(args.learning_rate) + '.txt'
    if not os.path.exists(args.logs_path + '/combine'):
        os.makedirs(args.logs_path + '/combine')

    if (os.path.isfile(angle_file)):
        os.remove(angle_file)  # 删掉已有同名文件
    f_angle = open(angle_file, 'a')

    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        B = label.shape[0]

        optimizer_a.zero_grad()
        optimizer_v.zero_grad()

        out_a = audio_net(spec.unsqueeze(1).float())

        out_v = visual_net(image.float(), B)

        if args.fusion_method == 'sum':
            out = out_a + out_v

        loss = criterion(out, label)
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)
        print('loss: ', loss, loss_a, loss_v)

        grad_a = torch.Tensor([]).to(device)
        grad_v = torch.Tensor([]).to(device)
        grad_a_fusion = torch.Tensor([]).to(device)
        grad_v_fusion = torch.Tensor([]).to(device)
        loss_a.backward(retain_graph=True)
        for parms in audio_net.parameters():
            grad_a = torch.cat((grad_a, parms.grad.flatten()), 0)
        optimizer_a.zero_grad()

        loss_v.backward(retain_graph=True)
        for parms in visual_net.parameters():
            grad_v = torch.cat((grad_v, parms.grad.flatten()), 0)
        optimizer_v.zero_grad()

        loss.backward()
        for parms in audio_net.parameters():
            grad_a_fusion = torch.cat((grad_a_fusion, parms.grad.flatten()), 0)
        for parms in visual_net.parameters():
            grad_v_fusion = torch.cat((grad_v_fusion, parms.grad.flatten()), 0)

        _, a_angle = dot_product_angle_tensor(grad_a, grad_a_fusion)
        _, v_angle = dot_product_angle_tensor(grad_v, grad_v_fusion)
        _a_angle += a_angle
        _v_angle += v_angle
        print('angle', a_angle, v_angle)

        f_angle.write(str(epoch) +
                      "\t" + str(a_angle) +
                      "\t" + str(v_angle) +
                      "\n")
        f_angle.flush()

        optimizer_a.step()
        optimizer_v.step()

        _loss += loss.item()
    scheduler.step()
    f_angle.close()
    return _loss / len(dataloader), _a_angle, _v_angle


def valid_uniclassifier(args, audio_net, visual_net, device, dataloader):
    audio_net.eval()
    visual_net.eval()

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

            out_a = audio_net(spec.unsqueeze(1).float())

            out_v = visual_net(image.float(), B)

            if args.fusion_method == 'sum':
                out = out_a + out_v

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

    audio_net = AClassifier(args)
    visual_net = VClassifier(args)

    audio_net.apply(weight_init)
    audio_net.to(device)
    visual_net.apply(weight_init)
    visual_net.to(device)

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
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:
        trainloss_file = args.logs_path + '/combine' + '/classifier_train_loss-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) + '.txt'
        if not os.path.exists(args.logs_path + '/combine'):
            os.makedirs(args.logs_path + '/combine')

        save_path = args.ckpt_path + '/combine' + '/classifier-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                    str(args.batch_size) + '-lr' + str(args.learning_rate)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        # load trained encoder

        load_path_a = args.load_path_a
        load_dict_a = torch.load(load_path_a)
        state_dict_a = load_dict_a['model']
        audio_net.load_state_dict(state_dict_a)

        load_path_v = args.load_path_v
        load_dict_v = torch.load(load_path_v)
        state_dict_v = load_dict_v['model']
        visual_net.load_state_dict(state_dict_v)

        optimizer_a = optim.SGD(audio_net.parameters(), lr=args.learning_rate, momentum=0.9,
                                weight_decay=1e-4)
        optimizer_v = optim.SGD(visual_net.parameters(), lr=args.learning_rate, momentum=0.9,
                                weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer_a, args.lr_decay_step, args.lr_decay_ratio)

        best_acc = 0
        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))

            batch_loss, a_angle, v_angle = train_uniclassifier_epoch(args, epoch, audio_net, visual_net, device,
                                                                     train_dataloader, optimizer_a, optimizer_v,
                                                                     scheduler)
            acc = valid_uniclassifier(args, audio_net, visual_net, device, test_dataloader)
            print('epoch: ', epoch, 'loss: ', batch_loss, 'acc: ', acc)

            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(acc) +
                              "\t" + str(a_angle) +
                              "\t" + str(v_angle) +
                              "\n")
            f_trainloss.flush()

            if acc > best_acc or (epoch + 1) % args.epochs == 0:
                if acc > best_acc:
                    best_acc = float(acc)
                print('Saving model....')
                # torch.save(
                #     {
                #         'classifier': classifier.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'scheduler': scheduler.state_dict()
                #     },
                #     os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
                # )
                print('Saved model!!!')

        f_trainloss.close()


if __name__ == '__main__':
    main()
