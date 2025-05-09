import torch
import torch.backends.cudnn as cudnn

import argparse
import logging
import os
import random
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.ori_vit_seg_modeling import VisionTransformer as ViT_seg_ori
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/downstream/CVC-ClinicVideoDB/', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=2e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=2, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--inference', action='store_true', help='inference the seg results')
    parser.add_argument('--pretrained_model_weights', type=str, default='cvc.pth', help='pretrained weights')
    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/downstream/CVC-ClinicVideoDB/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net = ViT_seg_ori(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net.load_from(weights=np.load(config_vit.pretrained_path))

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)

    # seed 0: 82.2
    # seed 1: 83.9
    # seed 2: 83.8

    # Endo-LV 1min 32frame 2141clips
    # seed 0: 83.6
    # seed 1: 83.8
    # seed 2: 83.0

    # Endo-LV 20min
    # seed 0: 81.7
    # seed 1: 83.4
    # seed 2: 84.1

    # EndoSSL
    # seed 0:
    # seed 1:
    # seed 2:
