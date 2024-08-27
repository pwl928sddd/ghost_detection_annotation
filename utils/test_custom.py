from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
sys.path.append('..')
from pointnet.model_custom import PointNetCls, PointNetDenseCls, feature_transform_regularizer
from pointnet.my_dataset import myDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math


torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset_path', type=str, default='/home/mahapatro/pointnet.pytorch/custom_data/', help='dataset path')

opt = parser.parse_args()
print(opt)
opt.model = 'cls/seg_model_1_99.pth'
blue = lambda x: '\033[94m' + x + '\033[0m'

opt.class_choice = 1

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

batch_size = 32

test_dataset = myDataset(root_dir = opt.dataset_path, split='test')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_classes = 1
print('classes', num_classes)

classifier = PointNetDenseCls(k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

classifier.cuda()

num_batch = len(test_dataset) / batch_size

test_loss = 0
test_accuracy = 0
test_correct = 0

num_seen = 0

for i,data in enumerate(test_loader, 0):
    points, target, real_len, filepath = data
    points = points.transpose(2, 1)
    points, target = points.float().cuda(), target.float().cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_real = torch.empty(0, ).cuda()
    target_real = torch.empty(0, ).cuda()
    for j in range(len(pred)):
        x_p = pred[j][0:real_len[j],:].view(-1)
        x_t = target[j][0:real_len[j]].view(-1)
        pred_real = torch.cat((pred_real, x_p), 0)
        target_real = torch.cat((target_real, x_t), 0)
        x_p = x_p.unsqueeze(1)
        x_p[x_p > 0] = 1
        x_p[x_p <= 0] = 0
        x_t = x_t.unsqueeze(1)
        predicted_data = torch.cat((points[j].transpose(0,1)[0:real_len[j]], x_p, x_t),1).detach().cpu().numpy()
        # 如果没有os.path.join(opt.dataset_path, 'predicted_data')文件夹，则创建
        if not os.path.exists(os.path.join(opt.dataset_path, 'predicted_data')):
            os.makedirs(os.path.join(opt.dataset_path, 'predicted_data'))
        
        if "aug" not in filepath[j]:
            # import pdb; pdb.set_trace()
            np.savetxt(os.path.join(opt.dataset_path, 'predicted_data', filepath[j][-10:-4]+'_predicted.txt'), predicted_data, header="X Y Z V_r Mag Label Pred", delimiter=" ", comments='')
        else:
            np.savetxt(os.path.join(opt.dataset_path, 'predicted_data', filepath[j][-14:-4]+'_predicted.txt'), predicted_data, header="X Y Z V_r Mag Label Pred", delimiter=" ", comments='')
    loss = F.binary_cross_entropy_with_logits(pred_real, target_real)
    pred_choice = pred_real.data

    pred_choice[pred_choice > 0] = 1
    pred_choice[pred_choice <= 0] = 0
    correct = pred_choice.eq(target_real.data).cpu().sum()

    test_loss += loss.detach().cpu().numpy()
    test_correct += correct.item()

    num_seen += len(target_real)
    print('[%d/%d] train loss: %f accuracy: %f' % (i, num_batch, loss.detach().cpu().numpy(), correct.item() / float(torch.sum(real_len))))

print("Loss/test", test_loss / len(test_loader))
print("Accuracy/test", test_correct / float(num_seen))
