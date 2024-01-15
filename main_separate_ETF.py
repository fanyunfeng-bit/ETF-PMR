import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from dataset.CGMNIST import CGMNISTDataset
from dataset.CramedDataset import CramedDataset
from dataset.AVEDataset import AVEDataset
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, CGClassifier
from utils.ETF_utils import get_global_etf, update_global_etf, update_separate_etf, save_separate_feature
from utils.utils import setup_seed, weight_init

from dataset.VGGSoundDataset import VGGSound
import time
import matlab.engine
import scipy.io as scio


etf_file_a = "feature_matrix/ETF_a_matrix_norm.mat"
etf_file_v = "feature_matrix/ETF_v_matrix_norm.mat"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'Acc', 'Proto', 'ETF'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--optimizer', default='SGD', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--momentum_coef', default=0.2, type=float)
    parser.add_argument('--proto_update_freq', default=50, type=int, help='steps')

    # parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=70, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in Proto')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument('--warmup_epoch', default=0, type=int, help='where modulation begins')

    parser.add_argument('--class_imbalanced', action='store_true')

    return parser.parse_args()


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def grad_amplitude_diff(v1, v2):
    len_v1 = torch.norm(v1, p=2)
    len_v2 = torch.norm(v2, p=2)
    return len_v1, len_v2, len_v1 - len_v2


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, ETF_a_proto, ETF_v_proto, warmup=True):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_etf = 0

    for step, (spec, image, label) in enumerate(dataloader):

        spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
        image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
        label = label.to(device)  # B

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.dataset != 'CGMNIST':
            a, v, out = model(spec.unsqueeze(1).float(), image.float())
        else:
            a, v, out = model(spec, image)  # gray colored

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                     model.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                     model.fusion_module.fc_x.bias)
        elif args.fusion_method == 'concat':
            weight_size = model.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
        elif args.fusion_method == 'film':
            out_v = out
            out_a = out
        elif args.fusion_method == 'gated':
            out_v = out
            out_a = out

        loss = criterion(out, label)
        loss_etf = 0
        if args.modulation_starts <= epoch < args.modulation_ends:
            # loss_a = 0
            # loss_v = 0
            # for i in range(label.shape[0]):
            #     a_etf = ETF_a_proto[label[i]]
            #     v_etf = ETF_v_proto[label[i]]
            #     distance_a = (torch.dist(a[i], a_etf, p=2)) ** 2 / label.shape[0]
            #     distance_v = (torch.dist(v[i], v_etf, p=2)) ** 2 / label.shape[0]
            #     assert (torch.isnan(distance_a).sum() == 0)
            #     assert (torch.isnan(distance_v).sum() == 0)
            #     loss_a += distance_a
            #     loss_v += distance_v

            audio_sim = -EU_dist(a, ETF_a_proto)  # B x n_class
            visual_sim = -EU_dist(v, ETF_v_proto)  # B x n_class

            loss_a = criterion(audio_sim, label)
            loss_v = criterion(visual_sim, label)

            if not warmup:
                loss_etf = loss_a + loss_v
                loss += args.alpha * loss_etf

        loss.backward()

        optimizer.step()

        _loss += loss.item()
        if loss_etf != 0:
            _loss_etf += loss_etf.item()

    if args.optimizer == 'SGD':
        scheduler.step()
    # f_angle.close()

    return _loss / len(dataloader), _loss_etf / len(dataloader)


def valid(args, model, device, dataloader, audio_proto, visual_proto):
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
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        acc_a_p = [0.0 for _ in range(n_classes)]
        acc_v_p = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
            elif args.fusion_method == 'film':
                out_v = out
                out_a = out
            elif args.fusion_method == 'gated':
                out_v = out
                out_a = out

            prediction = softmax(out)
            # pred_v = softmax(out_v)
            # pred_a = softmax(out_a)

            audio_sim = -EU_dist(a, audio_proto)  # B x n_class
            visual_sim = -EU_dist(v, visual_proto)  # B x n_class
            # print(audio_sim, visual_sim, (audio_sim != audio_sim).any(), (visual_sim != visual_sim).any())
            pred_v_p = softmax(visual_sim)
            pred_a_p = softmax(audio_sim)
            # print('pred_p: ', (pred_a_p != pred_a_p).any(), (pred_v_p != pred_v_p).any())

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                # v = np.argmax(pred_v[i].cpu().data.numpy())
                # a = np.argmax(pred_a[i].cpu().data.numpy())
                v_p = np.argmax(pred_v_p[i].cpu().data.numpy())
                a_p = np.argmax(pred_a_p[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                # if np.asarray(label[i].cpu()) == v:
                #     acc_v[label[i]] += 1.0
                # if np.asarray(label[i].cpu()) == a:
                #     acc_a[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v_p:
                    acc_v_p[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a_p:
                    acc_a_p[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a_p) / sum(num), sum(acc_v_p) / sum(num)


def calculate_prototype(args, model, dataloader, device, epoch, a_proto=None, v_proto=None):
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

    audio_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    visual_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)  # B x 257 x 1004
            image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
            label = label.to(device)  # B

            # TODO: make it simpler and easier to extend
            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            for c, l in enumerate(label):
                l = l.long()
                count_class[l] += 1
                audio_prototypes[l, :] += a[c, :]
                visual_prototypes[l, :] += v[c, :]

            sample_count += 1
            if args.dataset == 'AVE':
                pass
            else:
                if sample_count >= all_num // 10:
                    break
    for c in range(audio_prototypes.shape[0]):
        audio_prototypes[c, :] /= count_class[c]
        visual_prototypes[c, :] /= count_class[c]

    if epoch <= 0:
        audio_prototypes = audio_prototypes
        visual_prototypes = visual_prototypes
    else:
        audio_prototypes = (1 - args.momentum_coef) * audio_prototypes + args.momentum_coef * a_proto
        visual_prototypes = (1 - args.momentum_coef) * visual_prototypes + args.momentum_coef * v_proto
    return audio_prototypes, visual_prototypes


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.dataset == 'CGMNIST':
        model = CGClassifier(args)
    else:
        model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        scheduler = None

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

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train', class_imbalanced=args.class_imbalanced)
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train', class_imbalanced=args.class_imbalanced)
        test_dataset = AVEDataset(args, mode='test')
        val_dataset = AVEDataset(args, mode='val')
    elif args.dataset == 'CGMNIST':
        train_dataset = CGMNISTDataset(args, mode='train')
        test_dataset = CGMNISTDataset(args, mode='test')
        val_dataset = CGMNISTDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.dataset == 'AVE':
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, pin_memory=False)
    elif args.dataset == 'CGMNIST':
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=True, pin_memory=False)


    eng = matlab.engine.start_matlab()
    eng.cd(r'D:\MyResearch\Regions\Federated-Learning\Multi-modal-FL\code\Mine\PMR-ModalityImbalance\OptM-master', nargout=0)

    if args.train:

        trainloss_file = args.logs_path + '/ETF/' + args.dataset + '-Imbalanced-' + str(args.class_imbalanced) + '/train_log-ETFproto-update-alpha=3-.txt'
        if not os.path.exists(args.logs_path + '/ETF/' + args.dataset+ '-Imbalanced-' + str(args.class_imbalanced)):
            os.makedirs(args.logs_path + '/ETF/' + args.dataset+ '-Imbalanced-' + str(args.class_imbalanced))

        save_path = args.ckpt_path + '/ETF' + '/model-ETFproto-update-alpha=3-' + args.dataset + '-Imbalanced-' + str(args.class_imbalanced)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        best_acc = 0.0

        feature_size = args.embed_dim
        # ETF_a_proto = get_global_etf(feature_size, n_classes, pre_generated=True, file=etf_file_a).to(
        #     device)  # num_class x embed_dim
        # ETF_v_proto = get_global_etf(feature_size, n_classes, pre_generated=True, file=etf_file_v).to(
        #     device)  # num_class x embed_dim

        # ETF_a_proto, ETF_v_proto = calculate_prototype(args, model, train_dataloader, device, 0)

        # matlab
        a_feature_file = "feature_a_matrix-new.mat"
        a_ETF_file = "ETF_a_matrix-new.mat"
        v_feature_file = "feature_v_matrix-new.mat"
        v_ETF_file = "ETF_v_matrix-new.mat"
        audio_proto, visual_proto = save_separate_feature(args, train_dataloader, model, device)
        eng.optimalETF(a_feature_file, a_ETF_file)
        eng.optimalETF(v_feature_file, v_ETF_file)
        a_etf = scio.loadmat('OptM-master/'+a_ETF_file)['W_p']
        ETF_a_proto = torch.tensor(a_etf).T.to(device)
        v_etf = scio.loadmat('OptM-master/'+v_ETF_file)['W_p']
        ETF_v_proto = torch.tensor(v_etf).T.to(device)

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            s_time = time.time()

            if epoch < args.warmup_epoch:
                batch_loss, batch_loss_etf = train_epoch(args, epoch, model, device, train_dataloader, optimizer,
                                                         scheduler, ETF_a_proto, ETF_v_proto, warmup=True)
            else:
                batch_loss, batch_loss_etf = train_epoch(args, epoch, model, device, train_dataloader, optimizer,
                                                         scheduler, ETF_a_proto, ETF_v_proto, warmup=False)

            e_time = time.time()
            print('per epoch time: ', e_time - s_time)

            if epoch == args.warmup_epoch:
                # update ETF_proto according to current feature
                # ETF_a_proto, ETF_v_proto = update_separate_etf(args, ETF_a_proto, ETF_v_proto, train_dataloader, model,
                #                                                device, pre_generated=False, file_a=etf_file_a, file_v=etf_file_v)
                ETF_a_proto, ETF_v_proto = calculate_prototype(args, model, train_dataloader, device, 0)

            # if args.dataset == 'AVE':
            #     audio_proto, visual_proto = calculate_prototype(args, model, val_dataloader, device, 0)
            # elif args.dataset == 'CGMNIST':
            #     audio_proto, visual_proto = calculate_prototype(args, model, val_dataloader, device, 0)
            # else:
            #     audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, 0)

            audio_proto, visual_proto = save_separate_feature(args, train_dataloader, model, device)
            audio_proto = torch.tensor(audio_proto).T.to(device)
            visual_proto = torch.tensor(visual_proto).T.to(device)
            eng.optimalETF(a_feature_file, a_ETF_file)
            eng.optimalETF(v_feature_file, v_ETF_file)
            a_etf = scio.loadmat('OptM-master/'+a_ETF_file)['W_p']
            ETF_a_proto = torch.tensor(a_etf).T.to(device)
            v_etf = scio.loadmat('OptM-master/'+v_ETF_file)['W_p']
            ETF_v_proto = torch.tensor(v_etf).T.to(device)

            # ETF_a_proto, ETF_v_proto = audio_proto, visual_proto
            acc, acc_a_p, acc_v_p = valid(args, model, device, test_dataloader, audio_proto, visual_proto)
            print('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_etf)
            print('epoch: ', epoch, 'acc: ', acc, 'acc_v_p: ', acc_v_p, 'acc_a_p: ', acc_a_p)
            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(batch_loss_etf) +
                              "\t" + str(acc) +
                              "\t" + str(acc_a_p) +
                              "\t" + str(acc_v_p) +
                              "\n")
            f_trainloss.flush()

            # if acc > best_acc or (epoch + 1) % 10 == 0:
            #     if acc > best_acc:
            #         best_acc = float(acc)
            #
            #     print('Saving model....')
            #     torch.save(
            #         {
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict()
            #         },
            #         os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
            #     )
            #     print('Saved model!!!')
        f_trainloss.close()

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
