import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import os
from tqdm import tqdm
from einops import rearrange


parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=400, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=400, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--exp', default='memory_bank', type=str, help='exp name')
parser.add_argument('--test', action='store_true', help='test mode')

args = parser.parse_args()

gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.6f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))
print('expname         : {:s}'.format(args.exp))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self,degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img,brightness_factor)
        img_ = TF.adjust_contrast(img_,contrast_factor)
        img_ = TF.adjust_saturation(img_,saturation_factor)
        img_ = TF.adjust_hue(img_,hue_factor)
        
        return img_


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:,0]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        try:
            img_names = self.file_paths[index]
            labels_phase = self.file_labels_phase[index]
        except:
            print(len(self.file_paths), index)

        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase

    def __len__(self):
        return len(self.file_paths)

class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # torch.Size([10, 10, 3, 224, 224])
        # torch.Size([100, 3, 224, 224])
        # torch.Size([100, 2048, 1, 1])
        # torch.Size([10, 10, 2048])
        # torch.Size([10, 10, 512])
        # torch.Size([100, 512])
        # torch.Size([100, 512])
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc(y)
        return y


class vit_lstm(torch.nn.Module):
    def __init__(self):
        super(vit_lstm, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        from vit import get_vit_base_patch16_224
        body = get_vit_base_patch16_224()

        # weight = '../checkpoints/endolv_1min_32frame_3gpu/checkpoint.pth'
        # weight = '../checkpoints/endolv_1min_32frame_3565video_3gpu/checkpoint.pth'
        weight = '../checkpoints/endolv_1min_32frame_3565video_3gpu_new/checkpoint.pth'
        # weight = '../checkpoints/endo_fm.pth'
        # weight = '../checkpoints/endossl.pth'
        print(f'Loading {weight}...')
        ckpt = torch.load(weight, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]

        # print(ckpt.keys())
        if 'TimeSformer' in weight:
            ckpt = {"backbone." + key[len("model."):]: value for key, value in ckpt.items()}
        # print(ckpt.keys())

        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = body.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")

        self.layer = body.blocks
        self.encoder_norm = body.norm
        self.cls_token = body.cls_token

        # print(self.cls_token.shape); exit(0)  # torch.Size([1, 1, 768])

        self.projection = nn.Conv2d(2048, 768, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.lstm = nn.LSTM(768, 768, batch_first=True)
        self.fc = nn.Linear(768, 7)
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            x = x.view(-1, 3, 224, 224)
            x = self.share.forward(x)  # [100, 2048, 7, 7]
            H, W = x.shape[-2], x.shape[-1]
            # print(x.shape)
            x = self.projection(x).view(-1, sequence_length, 768, H, W)
            # print(x.shape)

            x = self.recover_for_trans(x)
            # print(x.shape)
            x = torch.cat([self.cls_token.expand((x.shape[0], 1, self.cls_token.shape[-1])), x], dim=1)
            # print(x.shape)

            for layer_block in self.layer:
                x = layer_block(x, B=x.shape[0], T=sequence_length, W=W)
            # print(x.shape)

            x = self.encoder_norm(x)
            x = self.recover_for_conv(x[:, 1:, :], T=sequence_length, W=W)
            # print(x.shape)

            x = self.avgpool(x).view(-1, sequence_length, 768)
            # print(x.shape)

            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
            # print(x.shape)
            x = x.contiguous().view(-1, 768)
            x = self.dropout(x)
            x = self.fc(x)

            # exit(0)
        return x

    def recover_for_trans(self, x):
        B, T, D, H, W = x.shape
        # print(x.shape, cls_token.shape, T, H, W)
        x = rearrange(x, 'b t m h w -> b (h w t) m', b=B, m=D, h=H, w=W, t=T)
        return x

    def recover_for_conv(self, x, T, W):
        B = x.shape[0]
        num_spatial_tokens = x.shape[1] // T
        H = num_spatial_tokens // W
        D = x.shape[2]
        # print(x.shape, cls_token.shape, T, H, W)
        x = rearrange(x, 'b (h w t) m -> b t m h w', b=B, m=D, h=H, w=W, t=T)
        return x


class vit_lstm2(torch.nn.Module):
    def __init__(self):
        super(vit_lstm2, self).__init__()
        from vit import get_vit_base_patch16_224
        body = get_vit_base_patch16_224()

        weight = '../checkpoints/endolv_1min_32frame_3gpu/checkpoint.pth'
        print(f'Loading {weight}...')
        ckpt = torch.load(weight, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]

        # print(ckpt.keys())
        if 'TimeSformer' in weight:
            ckpt = {"backbone." + key[len("model."):]: value for key, value in ckpt.items()}
        # print(ckpt.keys())

        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = body.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")

        self.share = body
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.lstm = nn.LSTM(768, 768, batch_first=True)
        self.fc = nn.Linear(768, 7)
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            x = x.permute(0, 2, 1, 3, 4)

            x = self.share.forward(x)
            # x = self.avgpool(x).view(-1, sequence_length, 768)
            # self.lstm.flatten_parameters()
            # x, _ = self.lstm(x)
            # print(x.shape)
            x = x.contiguous().view(-1, 768)
            x = self.dropout(x)
            x = self.fc(x)
        return x


def get_useful_start_idx2(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        each_idx = []
        for j in range(count, count + (list_each_length[i] // sequence_length * sequence_length - sequence_length)):
            each_idx.append(j)
        count += len(each_idx)
        idx.append(each_idx)
    return idx


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    used_each_length = []
    for i in range(len(list_each_length)):
        each_count = 0
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
            each_count += 1
        count += list_each_length[i]
        used_each_length.append(each_count)
    return idx, used_each_length


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_19  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_19 : {:6d}'.format(len(val_labels_80)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])

    train_dataset_80 = CholecDataset(train_paths_80, train_labels_80, train_transforms)
    val_dataset_80 = CholecDataset(val_paths_80, val_labels_80, test_transforms)

    return train_dataset_80, train_num_each_80, \
           val_dataset_80, val_num_each_80


# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


sig_f = nn.Sigmoid()


def valMinibatch(testloader, model, val_used_each_length):
    each_start_end = [[sum(val_used_each_length[:i]), sum(val_used_each_length[:i + 1])]
                      for i in range(len(val_used_each_length))]
    # print(val_used_each_length)
    # print(each_start_end)
    # exit(0)

    model.eval()
    criterion_phase = nn.CrossEntropyLoss(size_average=False)
    with torch.no_grad():
        val_loss_phase = 0.0
        val_corrects_phase = 0.0

        val_all_preds_phase = []
        val_all_labels_phase = []
        for data in tqdm(testloader, desc='Testing', leave=False):
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            val_loss_phase += loss_phase.data.item()
            # val_corrects_phase += torch.sum(preds_phase == labels_phase.data)

            for i in range(len(preds_phase)):
                val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
            for i in range(len(labels_phase)):
                val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))

        val_all_preds_phase = torch.tensor(val_all_preds_phase)
        val_all_labels_phase = torch.tensor(val_all_labels_phase)

        all_acc = []
        for each_video in each_start_end:
            si, ei = each_video
            this_acc = torch.sum(val_all_preds_phase[si:ei] == val_all_labels_phase[si:ei]) / (ei -si)
            all_acc.append(this_acc)

        acc_mean = np.mean(np.array(all_acc))
        acc_std = np.std(np.array(all_acc))

    return acc_mean, acc_std, val_loss_phase / len(val_all_labels_phase)


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    # TensorBoard
    # writer = SummaryWriter('runs/lr5e-4_do/')

    (train_dataset_80),\
    (train_num_each_80),\
    (val_dataset),\
    (val_num_each) = train_dataset, train_num_each, val_dataset, val_num_each

    train_useful_start_idx_80, train_used_each_length = get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx, val_used_each_length = get_useful_start_idx(sequence_length, val_num_each)
    # train_useful_start_idx_80 = get_useful_start_idx2(sequence_length, train_num_each_80)
    # val_useful_start_idx = get_useful_start_idx2(sequence_length, val_num_each)

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)

    train_we_use_start_idx_80 = train_useful_start_idx_80
    val_we_use_start_idx = val_useful_start_idx

    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    # train_idx = []
    # for each_idx in train_we_use_start_idx_80:
    #     for i in range(len(each_idx) // sequence_length):
    #         for j in range(sequence_length):
    #             train_idx.append(each_idx[i] * sequence_length + j)
    #
    # val_idx = []
    # for each_idx in val_we_use_start_idx:
    #     for i in range(len(each_idx) // sequence_length):
    #         for j in range(sequence_length):
    #             val_idx.append(each_idx[i] * sequence_length + j)

    # print(train_we_use_start_idx_80[0], train_we_use_start_idx_80[-1])
    # print(val_we_use_start_idx[0], val_we_use_start_idx[-1])

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all valid use: {:6d}'.format(num_val_all))

    # print(val_num_each)
    # print(val_used_each_length, sum(val_used_each_length))
    # exit(0)

    # train_loader_80 = DataLoader(
    #     train_dataset_80,
    #     batch_size=train_batch_size,
    #     sampler=SeqSampler(train_dataset_80, train_idx),
    #     num_workers=workers,
    #     pin_memory=True
    # )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=True
    )

    # model = resnet_lstm()
    model = vit_lstm()


    if args.test:
        print(f'Loading output/{args.exp}/best.pth')
        save_dir = f'output/{args.exp}'
        msg = model.load_state_dict(torch.load(f'{save_dir}/best.pth'), strict=True)
        print(msg)

        if use_gpu:
            model = DataParallel(model)
            model.to(device)

        val_acc_mean, val_acc_std, val_loss_phase = valMinibatch(val_loader, model, val_used_each_length)

        print(f'Test acc {round(val_acc_mean * 100, 1)}+-{round(val_acc_std * 100, 1)}')

        exit(0)

       
    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    criterion_phase = nn.CrossEntropyLoss(size_average=False)

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': learning_rate}],
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epo, eta_min=0)

    # optimizer = None
    # exp_lr_scheduler = None
    #
    # if multi_optim == 0:
    #     if optimizer_choice == 0:
    #         optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
    #                               weight_decay=weight_decay, nesterov=use_nesterov)
    #         if sgd_adjust_lr == 0:
    #             exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
    #         elif sgd_adjust_lr == 1:
    #             exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #     elif optimizer_choice == 1:
    #         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # elif multi_optim == 1:
    #     if optimizer_choice == 0:
    #         optimizer = optim.SGD([
    #             {'params': model.module.share.parameters()},
    #             {'params': model.module.lstm.parameters(), 'lr': learning_rate},
    #             {'params': model.module.fc.parameters(), 'lr': learning_rate},
    #         ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
    #             weight_decay=weight_decay, nesterov=use_nesterov)
    #         if sgd_adjust_lr == 0:
    #             exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
    #         elif sgd_adjust_lr == 1:
    #             exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #     elif optimizer_choice == 1:
    #         optimizer = optim.Adam([
    #             {'params': model.module.share.parameters()},
    #             {'params': model.module.lstm.parameters(), 'lr': learning_rate},
    #             {'params': model.module.fc.parameters(), 'lr': learning_rate},
    #         ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    best_val_std_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx_80)
        train_idx_80 = []
        for i in range(num_train_we_use_80):
            for j in range(sequence_length):
                train_idx_80.append(train_we_use_start_idx_80[i] + j)

        train_loader_80 = DataLoader(
            train_dataset_80,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset_80, train_idx_80),
            num_workers=workers,
            pin_memory=True
        )

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        # batch_progress = 0.0
        # running_loss_phase = 0.0
        # minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for data in tqdm(train_loader_80, desc='Training', leave=False):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)

            with torch.cuda.amp.autocast(enabled=True):
                outputs_phase = model.forward(inputs)
                # print(inputs.shape, outputs_phase.shape, labels_phase.shape); exit(0)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                loss = loss_phase

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            # running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            # minibatch_correct_phase += batch_corrects_phase


            # if i % 500 == 499:
            #     # ...log the running loss
            #     batch_iters = epoch * num_train_all/sequence_length + i*train_batch_size/sequence_length
            #     writer.add_scalar('training loss phase',
            #                       running_loss_phase / (train_batch_size*500/sequence_length) ,
            #                       batch_iters)
            #     # ...log the training acc
            #     writer.add_scalar('training acc phase',
            #                       float(minibatch_correct_phase) / (float(train_batch_size)*500/sequence_length),
            #                       batch_iters)
            #     # ...log the val acc loss
            #
            #     val_loss_phase, val_corrects_phase = valMinibatch(val_loader, model)
            #     writer.add_scalar('validation acc miniBatch phase',
            #                       float(val_corrects_phase) / float(num_val_we_use),
            #                       batch_iters)
            #     writer.add_scalar('validation loss miniBatch phase',
            #                       float(val_loss_phase) / float(num_val_we_use),
            #                       batch_iters)
            #
            #     running_loss_phase = 0.0
            #     minibatch_correct_phase = 0.0
            #
            # if (i+1)*train_batch_size >= num_train_all:
            #     running_loss_phase = 0.0
            #     minibatch_correct_phase = 0.0

            # batch_progress += 1
            # if batch_progress*train_batch_size >= num_train_all:
            #     percent = 100.0
            #     print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            # else:
            #     percent = round(batch_progress*train_batch_size / num_train_all * 100, 2)
            #     print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress*train_batch_size, num_train_all), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / float(num_train_all) * sequence_length
        train_average_loss_phase = train_loss_phase / num_train_all * sequence_length

        # print(train_accuracy_phase); exit(0)

        # Sets the module in evaluation mode.
        # model.eval()
        # val_loss_phase = 0.0
        # val_corrects_phase = 0
        val_start_time = time.time()
        # val_progress = 0
        # val_all_preds_phase = []
        # val_all_labels_phase = []

        # with torch.no_grad():
        #     for data in tqdm(val_loader, desc='Validating', leave=False):
        #         if use_gpu:
        #             inputs, labels_phase = data[0].to(device), data[1].to(device)
        #         else:
        #             inputs, labels_phase = data[0], data[1]
        #
        #         labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
        #
        #         inputs = inputs.view(-1, sequence_length, 3, 224, 224)
        #         outputs_phase = model.forward(inputs)
        #         outputs_phase = outputs_phase[sequence_length - 1::sequence_length]
        #
        #         _, preds_phase = torch.max(outputs_phase.data, 1)
        #         loss_phase = criterion_phase(outputs_phase, labels_phase)
        #
        #         val_loss_phase += loss_phase.data.item()
        #
        #         val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
        #         # TODO
        #
        #         for i in range(len(preds_phase)):
        #             val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
        #         for i in range(len(labels_phase)):
        #             val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))


                # val_progress += 1
                # if val_progress*val_batch_size >= num_val_all:
                #     percent = 100.0
                #     print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                # else:
                #     percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                #     print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        val_acc_mean, val_acc_std, val_loss_phase = valMinibatch(val_loader, model, val_used_each_length)
        val_accuracy_phase = val_acc_mean
        val_average_loss_phase = val_loss_phase
        val_elapsed_time = time.time() - val_start_time
        # val_accuracy_phase = float(val_corrects_phase) / float(num_val_we_use)
        # val_average_loss_phase = val_loss_phase / num_val_we_use

        # val_recall_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        # val_precision_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        # val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        # val_precision_each_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average=None)
        # val_recall_each_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average=None)

        # writer.add_scalar('validation acc epoch phase',
        #                   float(val_accuracy_phase),epoch)
        # writer.add_scalar('validation loss epoch phase',
        #                   float(val_average_loss_phase),epoch)

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss(phase): {:.4f}'
              ' train accu(phase): {:.1f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid loss(phase): {:.4f}'
              ' valid accu(phase): {:.1f}+-{:.1f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss_phase,
                      train_accuracy_phase * 100.,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss_phase,
                      val_accuracy_phase * 100.,
                      val_acc_std * 100.
                      ))

        # print("val_precision_each_phase:", val_precision_each_phase)
        # print("val_recall_each_pha
        #
        #
        #
        #
        # se:", val_recall_each_phase)
        # print("val_precision_phase", val_precision_phase)
        # print("val_recall_phase", val_recall_phase)
        # print("val_jaccard_phase", val_jaccard_phase)

        # if optimizer_choice == 0:
        #     if sgd_adjust_lr == 0:
        #         exp_lr_scheduler.step()
        #     elif sgd_adjust_lr == 1:
        #         exp_lr_scheduler.step(val_average_loss_phase)
        scheduler.step()

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            best_val_std_phase = val_acc_std
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch


            save_dir = f'output/{args.exp}'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_model_wts, f'{save_dir}/best.pth')

        # torch.save(model.module.state_dict(), f'{save_dir}/latest.pth')
        print(f'Best train acc {round(correspond_train_acc_phase * 100, 1)} '
              f'val acc {round(best_val_accuracy_phase * 100, 1)}+-{round(best_val_std_phase * 100, 1)} at epoch {best_epoch}')


def main():
    train_dataset_80, train_num_each_80, \
    val_dataset_80, val_num_each_80 = get_data('./train_val_paths_labels.pkl')
    train_model((train_dataset_80),
                (train_num_each_80),
                (val_dataset_80),
                (val_num_each_80))


if __name__ == "__main__":
    main()

print('Done')
print()
