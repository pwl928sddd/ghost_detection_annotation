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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter()

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--dataset_path', type=str, default='/home/mahapatro/pointnet.pytorch/custom_data/', help='dataset path')

opt = parser.parse_args()
print(opt)
opt.model = ''
blue = lambda x: '\033[94m' + x + '\033[0m'

opt.class_choice = 1

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

batch_size = 32

dataset = myDataset(root_dir = opt.dataset_path, split='train')
test_dataset = myDataset(root_dir = opt.dataset_path, split='test')

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_classes = 1
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
    
classifier = PointNetDenseCls(k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
#writer.add_graph(classifier)

num_batch = len(dataset) / batch_size

train_loss = 0
all_train_losses = []
all_test_losses = []
test_loss = 0
train_accuracy = 0
test_accuracy = 0

train_correct = 0
test_correct = 0

avg_training_loss = 0
avg_training_acc = 0

avg_test_loss = 0
avg_test_acc = 0

num_seen = 0

lambda1 = 0.5

for epoch in range(opt.nepoch):

    train_loss = 0
    test_loss = 0
    train_accuracy = 0
    test_accuracy = 0

    train_correct = 0
    test_correct = 0

    num_seen = 0

    for i, data in enumerate(train_loader, 0):
        points, target, _, _ = data
        points = points.transpose(2, 1)
        points, target = points.float().cuda(), target.float().cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data
        pred_choice[pred_choice > 0] = 1
        pred_choice[pred_choice <= 0] = 0
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.detach().cpu().numpy(), correct.item() / float(batch_size * 2500)))
        if i < len(train_loader):
            train_loss += loss.detach().cpu().numpy()
            train_correct += correct.item()

            num_seen += (2500 * batch_size)
    
    print("Loss/train", train_loss / (len(train_loader)-1), epoch)
    print("Accuracy/train", train_correct / float(num_seen), epoch)
    all_train_losses.append(train_loss / (len(train_loader)-1))
    writer.add_scalar("Loss/train", train_loss / (len(train_loader)-1), epoch)
    writer.add_scalar("Accuracy/train", train_correct / float(num_seen), epoch)

    num_seen = 0
    for j, data in enumerate(test_loader, 0):
        points, target, real_len, _ = data
        points = points.transpose(2, 1)
        points, target = points.float().cuda(), target.float().cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_real = torch.empty(0, ).cuda()
        target_real = torch.empty(0, ).cuda()
        #loss = F.binary_cross_entropy_with_logits(pred, target)
        for j in range(len(pred)):
            x_p = pred[j][0:real_len[j],:].view(-1)
            x_t = target[j][0:real_len[j]].view(-1)
            pred_real = torch.cat((pred_real, x_p), 0)
            target_real = torch.cat((target_real, x_t), 0)
        loss = F.binary_cross_entropy_with_logits(pred_real, target_real)
        pred_choice = pred_real.data

        pred_choice[pred_choice > 0] = 1
        pred_choice[pred_choice <= 0] = 0
        correct = pred_choice.eq(target_real.data).cpu().sum()

        test_loss += loss.detach().cpu().numpy()
        test_correct += correct.item()

        num_seen += len(target_real)

    writer.add_scalar("Loss/test", test_loss / len(test_loader), epoch)
    writer.add_scalar("Accuracy/test", test_correct / float(num_seen), epoch)
    all_test_losses.append(test_loss / len(test_loader))
    print("Loss/test", test_loss / len(test_loader), epoch)
    print("Accuracy/test", test_correct / float(num_seen), epoch)

    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))


plt.plot(all_train_losses, label='Training Loss')
plt.plot(all_test_losses, label='Test Loss')
plt.legend(frameon=True)
plt.savefig('loss.png')
plt.show()
