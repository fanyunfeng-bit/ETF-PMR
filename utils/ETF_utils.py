import numpy as np
import torch
import scipy.io as scio
from scipy.io import savemat

from utils.utils import calculate_prototype
import torch.nn.functional as F


# etf_file_a = "feature_matrix/ETF_a_matrix.mat"
# etf_file_v = "feature_matrix/ETF_v_matrix.mat"


# generate U in ETF equal, U is a random semi-orthogonal matrix, U*U^T = I,
# the size of U is d * K, where d is the feature size and K is the class nums
def generate_random_orthogonal_matrix(feature_size: int, num_classes: int):
    a = np.random.random(size=(feature_size, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(
        torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
    return P


# ETF: G = U * sqrt(K/K-1) * (I - 1/K * one*one^T) , where one is a vector filled with 1
# the size of G is d * K, where d is the feature size and K is the class nums
def generate_global_etf(feature_size: int, num_classes: int) -> torch.Tensor:
    P = generate_random_orthogonal_matrix(feature_size, num_classes)
    I = torch.eye(num_classes)
    one = torch.ones(num_classes, num_classes)
    M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))
    return torch.tensor(M).T


def get_global_etf(feature_size: int, class_num: int, pre_generated=False, file=None) -> torch.Tensor:
    if pre_generated == True:
        global_etf = scio.loadmat(file)['W_p']
        assert (global_etf.shape[0] == feature_size)
        assert (global_etf.shape[1] == class_num)
        print(global_etf.shape)
        global_etf = torch.tensor(global_etf).T
    else:
        global_etf = generate_global_etf(feature_size, class_num)
    return global_etf


def update_global_etf(args, ETF_proto, dataloader, model, device):
    """
    :params ETF_proto: n_class x 2*embed_dim
    :params dataloader
    """

    # if args.dataset == 'VGGSound':
    #     n_classes = 309
    # elif args.dataset == 'KineticSound':
    #     n_classes = 31
    # elif args.dataset == 'CREMAD':
    #     n_classes = 6
    # elif args.dataset == 'AVE':
    #     n_classes = 28
    # elif args.dataset == 'CGMNIST':
    #     n_classes = 10
    # else:
    #     raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    center = torch.zeros(args.embed_dim).to(device)
    features = []
    num_count = 0
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            center += torch.sum(torch.cat((a, v), 0), dim=0)
            features.append(torch.cat((a, v), 0))
            num_count += label.shape[0]

    center /= 2*num_count  # center

    norm_length = 0
    for fea in features:
        for f in fea:
            norm_length += torch.sqrt(torch.sum((f-center)*(f-center)))
    norm_length /= 2*num_count

    # print(norm_length, center.shape)
    for i, p in enumerate(ETF_proto):
        n_p = p*norm_length + center
        ETF_proto[i, :] = n_p
    return ETF_proto


def update_modality_etf(args, ETF_proto_a, ETF_proto_v, train_dataloader, model, device, ratio):
    a_proto, v_proto = calculate_prototype(args, model, train_dataloader, device, ratio=ratio, a_proto=None, v_proto=None)
    ETF_proto_a = args.momentum_coef * ETF_proto_a + (1 - args.momentum_coef) * a_proto
    ETF_proto_v = args.momentum_coef * ETF_proto_v + (1 - args.momentum_coef) * v_proto
    return ETF_proto_a, ETF_proto_v, a_proto, v_proto


def update_modality_etf_norm(args, ETF_proto_a, ETF_proto_v, train_dataloader, model, device, ratio):
    a_proto, v_proto = calculate_prototype(args, model, train_dataloader, device, ratio=ratio, a_proto=None, v_proto=None)
    a_proto_ = F.normalize(a_proto, p=2, dim=1)
    v_proto_ = F.normalize(v_proto, p=2, dim=1)

    ETF_proto_a = args.momentum_coef * ETF_proto_a + (1 - args.momentum_coef) * a_proto_
    ETF_proto_v = args.momentum_coef * ETF_proto_v + (1 - args.momentum_coef) * v_proto_

    ETF_proto_a = F.normalize(ETF_proto_a, p=2, dim=1)
    ETF_proto_v = F.normalize(ETF_proto_v, p=2, dim=1)
    return ETF_proto_a, ETF_proto_v, a_proto, v_proto


def update_separate_etf(args, ETF_a_proto, ETF_v_proto, dataloader, model, device, pre_generated=False, file_a=None, file_v=None):
    """
    :params ETF_proto: n_class x 2*embed_dim
    :params dataloader
    """
    if pre_generated:
        a_etf = scio.loadmat(file_a)['W_p']
        ETF_a_proto = torch.tensor(a_etf).T.to(device)
        v_etf = scio.loadmat(file_v)['W_p']
        ETF_v_proto = torch.tensor(v_etf).T.to(device)
    else:
        center_a = torch.zeros(args.embed_dim).to(device)
        center_v = torch.zeros(args.embed_dim).to(device)
        features_a = []
        features_v = []
        num_count = 0
        with torch.no_grad():
            model.eval()
            # TODO: more flexible
            for step, (spec, image, label) in enumerate(dataloader):

                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                if args.dataset != 'CGMNIST':
                    a, v, out = model(spec.unsqueeze(1).float(), image.float())
                else:
                    a, v, out = model(spec, image)  # gray colored

                center_a += torch.sum(a, dim=0)
                center_v += torch.sum(v, dim=0)
                features_a.append(a)
                features_v.append(v)
                num_count += label.shape[0]

        center_a /= num_count  # center
        center_v /= num_count

        norm_length_a = 0
        for fea in features_a:
            for f in fea:
                norm_length_a += torch.sqrt(torch.sum((f-center_a)*(f-center_a)))
        norm_length_a /= num_count

        norm_length_v = 0
        for fea in features_v:
            for f in fea:
                norm_length_v += torch.sqrt(torch.sum((f - center_v) * (f - center_v)))
        norm_length_v /= num_count

        for i, p in enumerate(ETF_a_proto):
            n_p = p*norm_length_a + center_a
            ETF_a_proto[i, :] = n_p
        for i, p in enumerate(ETF_v_proto):
            n_p = p*norm_length_v + center_v
            ETF_v_proto[i, :] = n_p
    return ETF_a_proto, ETF_v_proto


def save_separate_feature(args, dataloader, model, device, a_feature_file, v_feature_file, mm_feature_file):
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

    features_a = np.zeros([args.embed_dim, n_classes])
    features_v = np.zeros([args.embed_dim, n_classes])
    num_count = [0 for _ in range(n_classes)]
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            for i in range(label.shape[0]):
                features_a[:, label[i]] += a.cpu().numpy()[i]
                features_v[:, label[i]] += v.cpu().numpy()[i]
                num_count[label[i]] += 1
            # if step == 0:
            #     features_a = a.T.cpu().numpy()
            #     features_v = v.T.cpu().numpy()
            # else:
            #     features_a = np.concatenate((features_a, a.T.cpu().numpy()), axis=1)
            #     features_v = np.concatenate((features_v, v.T.cpu().numpy()), axis=1)

    for n in range(n_classes):
        features_a[:, n] /= num_count[n]
        features_v[:, n] /= num_count[n]
    # print(features_a.shape, features_v.shape, num_count)
    features_mm = (features_a + features_v) / 2

    savemat("OptM-master/{}".format(a_feature_file), {'feature': features_a})
    savemat("OptM-master/{}".format(v_feature_file), {'feature': features_v})
    savemat("OptM-master/{}".format(mm_feature_file), {'feature': features_mm})
    return features_a, features_v


# def ETF_solver(args, proto_a, proto_v, ):
#     if args.dataset == 'VGGSound':
#         n_classes = 309
#     elif args.dataset == 'KineticSound':
#         n_classes = 31
#     elif args.dataset == 'CREMAD':
#         n_classes = 6
#     elif args.dataset == 'AVE':
#         n_classes = 28
#     elif args.dataset == 'CGMNIST':
#         n_classes = 10
#     else:
#         raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
#
#     feature_size = args.embed_dim
#
#     for i in range(n_classes):