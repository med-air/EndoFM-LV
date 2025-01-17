import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import csv
import glob


datadir = 'pretrain/'

videolist = glob.glob(f'{datadir}/private2/*.mp4')
videolist.extend(glob.glob(f'{datadir}/public/*/*.mp4'))
videolist.extend(glob.glob(f'{datadir}/private_new/*.mp4'))
videolist.extend(glob.glob(f'{datadir}/private_batch3/*.mp4'))


total_frame = 0
total_duration = 0
video_count = 0
videos = []
for video in tqdm(videolist):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened(): continue

    fps, total_length = int(cap.get(5)), int(cap.get(7))
    # if total_length >= fps * 60 * 20:
    if True:
        video_count += 1
        total_frame += total_length
        duration = total_length /fps
        total_duration += duration
        videos.append([f'data/{video}', -1])

        # success, frame = cap.read()
        #
        # i = 1
        # while success:
        #     index = 4000
        #     if i > index:
        #         video_name = video.split('/')[-1].split('.')[0]
        #         os.makedirs(f'showcase/{video_name}', exist_ok=True)
        #         cv2.imwrite(os.path.join(f'showcase/{video_name}/frame{i}.jpg'), frame)
        #
        #         if i > (index + 20): break
        #
        #     i = i + 1
        #     success, frame = cap.read()

# 2141 4434917 41.06404629629635
print(len(videos), total_frame, total_duration / len(videos))

with open(f"{datadir}/train.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(videos)
