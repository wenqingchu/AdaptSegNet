import os
import shutil
f = open('val.txt')
for line in f.readlines():
    line = line.strip()
    source_file = '/home/chuwenqing/project/AdaptSegNet/data/GTA5/color_labels/' + line
    target_file = '/home/chuwenqing/project/AdaptSegNet/data/GTA5/val/color_labels/' + line
    shutil.copy(source_file, target_file)

    source_file = '/home/chuwenqing/project/AdaptSegNet/data/GTA5/labels/' + line

    target_file = '/home/chuwenqing/project/AdaptSegNet/data/GTA5/val/labels/' + line
    shutil.copy(source_file, target_file)

