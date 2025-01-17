import os
import cv2
cv2.setNumThreads(1)


def save_txt(data, file):
    f = open(file, "w")
    for line in data:
        f.write(line + '\n')
    f.close()


def read_txt(file):
    tmp = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            tmp.append(line)

    return tmp


total_video = 0
total_frame = 0
datadir = 'data/downstream/CRC/videos'
# name2label = {'adenoma': 0, 'advanced_CRC': 1, 'early_CRC': 2, 'hyperplastic_polyp': 3, 'SSL': 4}
name2label = {'adenoma': 0, 'advanced_CRC': 1, 'early_CRC': 0, 'hyperplastic_polyp': 2, 'SSL': 3}
# name2label = {'advanced_CRC': 0, 'early_CRC': 1, 'hyperplastic_polyp': 2, 'SSL': 3}
# [[40.  5.  2.  5.]
#  [ 4. 28.  1.  1.]
#  [25.  0.  9.  1.]
#  [ 8.  2.  3. 17.]]

train_data = []
val_data = []

folders = os.listdir(datadir)
for folder in folders:
    videos = os.listdir(os.path.join(datadir, folder))
    video_count = len(videos)

    print(folder, video_count)

    start_index = int(video_count * 0.)
    end_index = int(video_count * 0.2)

    for i in range(video_count):
        if videos[i] in ['.DS_Store']:
            print(videos[i])
            continue
        if folder in ['adenoma']:
            continue

        cap = cv2.VideoCapture(f'{datadir}/{folder}/{videos[i]}')
        total_frame += cap.get(7)
        total_video += video_count
        cap.release()

        item = f'{folder}/{videos[i]},{name2label[folder]}'

        if start_index < i < end_index: val_data.append(item)
        else: train_data.append(item)

save_txt(train_data, f'data/downstream/CRC/splits/train.txt')
save_txt(val_data, f'data/downstream/CRC/splits/val.txt')

print(len(train_data), len(val_data))
print(total_video, total_frame)
