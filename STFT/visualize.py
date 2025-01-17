import argparse
import os

import cv2
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import copy
import numpy as np


def get_colors():
    return [(111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (250, 170, 30),
            (220, 220, 0), (107, 142, 35),
            (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
            (0, 0, 90),
            (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32)]


def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--method', type=str, default='endofm')
    parser.add_argument('--datadir', type=str, default='../data/downstream/KUMC/processed/val2019/Image')

    return parser


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def convert_yolo2voc(image, bbx):
    height, width, channels = image.shape

    return int(bbx[0] * width - bbx[2] * width / 2), int(bbx[1] * height - bbx[3] * height / 2), \
           int(bbx[0] * width + bbx[2] * width / 2), int(bbx[1] * height + bbx[3] * height / 2)


def detect(image, category, four_axes, color, score=None):
    # rectangle
    cv2.rectangle(image, (int(four_axes[0]), int(four_axes[1])), (int(four_axes[2]), int(four_axes[3])), color, 4)
    # text
    font = cv2.FONT_HERSHEY_SIMPLEX
    categories = {'1': 'adenomatous', '2': 'hyperplastic'}
    text = categories[str(category)]

    if score is not None:
        cv2.putText(image, text + ' ' + str('{:.0f}'.format(score * 100)) + '%',
                    (int(four_axes[0]), int(four_axes[1]) - 4), font, 1, color, 3)
    else:
        cv2.putText(image, text, (int(four_axes[0]), int(four_axes[1]) - 4), font, 1, color, 3)

    return image


# class_to_id = {'adenomatous': 1, 'hyperplastic': 2}
def process_single(args, i, filename_lists, gt_boxlists, pred_boxlists, colors, vis_thr=0.5):
    filename = filename_lists[i]
    gt_boxlist = gt_boxlists[i]
    pred_boxlist = pred_boxlists[i]

    gt_bbox = gt_boxlist.bbox.numpy()
    gt_label = gt_boxlist.get_field("labels").numpy()

    pred_score = pred_boxlist.get_field("scores").numpy()
    # print(pred_score.shape)
    pred_bbox = pred_boxlist.bbox.numpy()
    # det_inds = pred_score >= vis_thr
    det_inds = [np.argmax(pred_score)]
    pred_score = pred_score[det_inds]
    pred_bbox = pred_bbox[det_inds]

    # print(det_inds); exit(0)

    if len(gt_label) > 0:
        image = cv2.imread(os.path.join(args.datadir, filename + '.jpg'))

        if len(pred_bbox) > 0:
            for bbox, score in zip(pred_bbox, pred_score):
                pred_image = copy.deepcopy(image)
                pred_image = detect(pred_image, gt_label[0], bbox, colors[gt_label[0]], score=score)
                cv2.imwrite(os.path.join('pred', args.method, filename.replace('/', '-') + '.jpg'), pred_image)

        # if len(gt_bbox) > 0:
        #     for bbox in gt_bbox:
        #         if not os.path.exists(os.path.join('pred', 'gt', filename + '.jpg')):
        #             gt_image = copy.deepcopy(image)
        #             gt_image = detect(gt_image, gt_label[0], bbox, colors[gt_label[0]])
        #             cv2.imwrite(os.path.join('pred', 'gt', filename.replace('/', '-') + '.jpg'), gt_image)

        # if not os.path.exists(os.path.join('pred', 'original', filename + '.jpg')):
        #     cv2.imwrite(os.path.join('pred', 'original', filename.replace('/', '-') + '.jpg'), image)


if __name__ == '__main__':
    args = get_parser().parse_args()
    args.rundir = os.path.join('pred', args.method)

    mkdir(args.rundir)
    mkdir(os.path.join('pred', 'gt'))
    mkdir(os.path.join('pred', 'original'))

    n_jobs = 10

    colors = get_colors()

    filename_lists = torch.load(os.path.join('saved', args.method, 'inference/CVCVideo_val_videos/filename_lists.pth'))
    pred_boxlists = torch.load(os.path.join('saved', args.method, 'inference/CVCVideo_val_videos/pred_boxlists.pth'))
    gt_boxlists = torch.load(os.path.join('saved', args.method, 'inference/CVCVideo_val_videos/gt_boxlists.pth'))

    for i in tqdm(range(len(filename_lists))):
        process_single(args, i, filename_lists, gt_boxlists, pred_boxlists, colors)

    # Parallel(n_jobs=n_jobs)(delayed(process_single)(args, i, filename_lists, gt_boxlists, pred_boxlists, colors)
    #                         for i in tqdm(range(len(filename_lists))))
