# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance, extract_features_classification
from evaluations import Recall_at_ks, NMI, Recall_at_ks_products
# import DataSet
import os
import numpy as np
from utils import to_numpy
# import pdb
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from CIFAR100 import CIFAR100
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('logs')

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cifar100')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument('-epochs', default=600, type=int, metavar='N', help='epochs for training process')
parser.add_argument('-num_task', type=int, default=2, help="learning rate of new parameters")
parser.add_argument('-nb_cl_fg', type=int, default=50, help="learning rate of new parameters")

parser.add_argument('-num_class', type=int, default=100, help="learning rate of new parameters")
parser.add_argument('-dir', default='/data/datasets/featureGeneration/',
                        help='data dir')
parser.add_argument('-top5', action = 'store_true', help='output top5 accuracy')

args = parser.parse_args()
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models = []
for i in os.listdir(args.r):
    if i.endswith("%d_model.pkl" % (args.epochs - 1)):  # 500_model.pkl
        models.append(os.path.join(args.r, i))

models.sort()

if 'cifar' in args.data:
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    testdir = args.dir + '/cifar'

if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            #transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        testdir = os.path.join(args.dir, 'ILSVRC12_256', 'val')

num_classes = args.num_class
num_task = args.num_task
num_class_per_task = (num_classes -  args.nb_cl_fg) // num_task

np.random.seed(args.seed)
#random_perm = np.random.permutation(num_classes)
random_perm = list(range(num_classes))

print('Test starting -->\t')
acc_all = np.zeros((num_task+3, num_task+1), dtype = 'float') # Save for csv

for task_id in range(num_task+1):
    if task_id ==0:
        index = random_perm[:args.nb_cl_fg]
    else:
        index = random_perm[:args.nb_cl_fg + (task_id) * num_class_per_task]
    if 'imagenet' in args.data:
        testfolder = ImageFolder(testdir, transform_test, index=index)
        test_loader = torch.utils.data.DataLoader(testfolder, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
    elif args.data =='cifar100':
        np.random.seed(args.seed)
        target_transform = np.random.permutation(num_classes)
        testset = CIFAR100(root=testdir, train=False, download=True, transform=transform_test, target_transform = target_transform, index = index)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2,drop_last=False)

    print('Test %d\t' % task_id)

    #model_loop = list(range(task_id, task_id + 1))
    #model_id = model_loop[0]
    model_id = task_id
    model = torch.load(models[model_id])


    val_embeddings_cl, val_labels_cl = extract_features_classification(model, test_loader, print_freq=32, metric=None)
    # Unknown task ID

    num_class = 0
    ave = 0.0
    weighted_ave = 0.0
    for k in range(task_id + 1):
        if k==0:
            tmp = random_perm[:args.nb_cl_fg]
        else:
            tmp = random_perm[args.nb_cl_fg + (k-1) * num_class_per_task:args.nb_cl_fg + (k) * num_class_per_task]
        gt = np.isin(val_labels_cl, tmp)
        if args.top5:
            estimate = np.argsort(val_embeddings_cl, axis=1)[:,-5:]
            estimate_label = estimate
            estimate_tmp = np.asarray(estimate_label)[gt]
            labels_tmp = np.tile(val_labels_cl[gt].reshape([len(val_labels_cl[gt]),1]),[1,5])
            acc = np.sum(estimate_tmp == labels_tmp) / float(len(estimate_tmp))
        else:
            estimate = np.argmax(val_embeddings_cl, axis=1)
            estimate_label = estimate
            estimate_tmp = np.asarray(estimate_label)[gt]
            acc = np.sum(estimate_tmp == val_labels_cl[gt]) / float(len(estimate_tmp))
        ave += acc
        weighted_ave += acc * len(tmp)
        num_class += len(tmp)
        print("Accuracy of Model %d on Task %d with unknown task boundary is %.3f" % (model_id, k, acc))
        acc_all[k, task_id] = acc
    print('Average: %.3f      Weighted Average: %.3f' %(ave / (task_id + 1), weighted_ave / num_class))
    acc_all[num_task + 1, task_id] = ave / (task_id + 1)
    acc_all[num_task + 2, task_id] = weighted_ave / num_class
    
if not os.path.exists('results/csv/'):
    os.makedirs('results/csv/')    
np.savetxt('results/csv/' + args.name + '.csv', acc_all*100, delimiter=",")