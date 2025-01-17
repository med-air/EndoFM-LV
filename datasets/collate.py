# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, n_tokens=None, mask_generator=None):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    # for j, s in enumerate(samples_list):
    #     for i in range(n_global_crops):
    #         print(j, i, s[0]["global_crops"][i].shape)
    # exit(0)

    # collated_global_crops = []
    # collated_local_crops = []

    # [collated_global_crops.append(s[0]["global_crops"][i]) for i in range(n_global_crops) for s in samples_list]
    # [collated_local_crops.append(s[0]["local_crops"][i]) for i in range(n_local_crops) for s in samples_list]

    collated_global_crops = [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    collated_local_crops = [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list]

    # print(samples_list[0][0].keys())
    # print(len(samples_list[0][0]["global_crops"]), samples_list[0][0]["global_crops"][0].shape)
    # print(len(samples_list[0][0]["local_crops"]), samples_list[0][0]["local_crops"][0].shape)
    # exit(0)

    B = len(collated_global_crops)
    # print(B, mask_probability);exit(0)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        # "collated_all_crops": collated_all_crops,
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }, 0, 0, 0
