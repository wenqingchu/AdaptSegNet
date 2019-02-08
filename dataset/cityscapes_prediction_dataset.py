import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, num_classes, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.label2train=[[0, 255],[1, 255],[2, 255],[3, 255],[4, 255],[5, 255],[6, 255],[7, 0],[8, 1],[9, 255],[10, 255],[11, 2],[12, 3],
                [13, 4],[14, 255],[15, 255],[16, 255],[17, 5],[18, 255],[19, 6],[20, 7],[21, 8],[22, 9],[23, 10],[24, 11],[25, 12],
                [26, 13],[27, 14],[28, 15],[29, 255],[30, 255],[31, 16],[32, 17],[33, 18],[-1, 255]]
        #self.id_to_trainid = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3,
        #             13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        #             26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255}

        #self.id_to_trainid = {0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 199, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3,
         #            13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
          #           26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18, -1: 19}


        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name[:-15] + "gtFine_labelIds.png"))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        #image = Image.open(datafiles["img"]).convert('RGB')
        #label = Image.open(datafiles["label"])
        image = np.load(datafiles["img"])
        name = datafiles["name"]

        # resize
        #image = cv2.resize(image, dsize=(self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_CUBIC)
        #image = image.resize(self.crop_size, Image.BICUBIC)
        #label = label.resize(self.crop_size, Image.NEAREST)

        #image = np.asarray(image, np.float32)
        #label = np.asarray(label, np.float32)

        #label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        #for k, v in self.id_to_trainid.items():
        #    label_copy[label == k] = v

        #for ind in range(len(self.label2train)):
        #    label_copy[label == self.label2train[ind][0]] = self.label2train[ind][1]
        #label_copy[label_copy == 255] = 19
        #label_copy = label_copy[np.newaxis,:]
        size = image.shape
        #image = image[:, :, ::-1]  # change to BGR
        #image -= self.mean
        #image = image.transpose((2, 0, 1))

        return image.copy(), image.copy(), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
