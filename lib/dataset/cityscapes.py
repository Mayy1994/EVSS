import os
from pdb import Pdb
import sys
import cv2
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lib.dataset.utils import transform_im, randomcrop
from matplotlib import pyplot as plt



class cityscapes_video_dataset(Dataset):
    def __init__(self, data_path, gt_path, list_path, crop_size=(512, 1024)):
        self.data_path = data_path
        self.gt_path = gt_path
        self.get_list(list_path)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.gt_label_name)

    def __getitem__(self, idx):
        
        img_2 = cv2.imread(os.path.join(self.data_path, self.img_2_name[idx]))
        img_2 = transform_im(img_2)
        img_3 = cv2.imread(os.path.join(self.data_path, self.img_3_name[idx]))
        img_3 = transform_im(img_3)

        gt_label = cv2.imread(os.path.join(self.gt_path, self.gt_label_name[idx]), 0)
        gt_edge_label = self.generate_edge(gt_label.copy())


        if np.random.rand() < 0.5:
            img_2 = np.flip(img_2, axis=2)
            img_3 = np.flip(img_3, axis=2)
            gt_label = np.flip(gt_label, axis=1)
            gt_edge_label = np.flip(gt_edge_label, axis=1)


        if self.crop_size is not None:
            [img_2, img_3, gt_label, gt_edge_label] = randomcrop([img_2, img_3, gt_label, gt_edge_label], crop_size=self.crop_size)
        img_2 = torch.from_numpy(img_2.copy())
        img_3 = torch.from_numpy(img_3.copy())
        gt_label = torch.from_numpy(gt_label.astype(np.int64))

        gt_edge_label = torch.from_numpy(gt_edge_label.astype(np.int64))


        return [img_2, img_3], gt_label, gt_edge_label

    def get_list(self, list_path):
        self.img_2_name = []
        self.img_3_name = []
        self.gt_label_name = []

        with open(list_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # if i % 5 == 0:
            img_3_name, gt_label_name = line.split()
            img_3_id = int(img_3_name[-22:-16])

            for j in range(1, 10):
                if j <= 5: 
                    img_2_id = img_3_id - j
                else:
                    img_2_id = img_3_id + j - 4                
                
                img_2_name = img_3_name.replace('{:06d}_leftImg8bit.png'.format(img_3_id),
                                                '{:06d}_leftImg8bit.png'.format(img_2_id))

                self.img_2_name.append(img_2_name)
                self.img_3_name.append(img_3_name)
                self.gt_label_name.append(gt_label_name)

    def generate_edge(self, label, edge_width=1):
        h, w = label.shape
        edge = np.zeros(label.shape)

        # right
        edge_right = edge[1:h, :]
        edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
                & (label[:h - 1, :] != 255)] = 1

        # up
        edge_up = edge[:, :w - 1]
        edge_up[(label[:, :w - 1] != label[:, 1:w])
                & (label[:, :w - 1] != 255)
                & (label[:, 1:w] != 255)] = 1

        # upright
        edge_upright = edge[:h - 1, :w - 1]
        edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                    & (label[:h - 1, :w - 1] != 255)
                    & (label[1:h, 1:w] != 255)] = 1

        # bottomright
        edge_bottomright = edge[:h - 1, 1:w]
        edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                        & (label[:h - 1, 1:w] != 255)
                        & (label[1:h, :w - 1] != 255)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        edge = cv2.dilate(edge, kernel)

        label[label==5]=19      # pole
        label[label==11]=19      # pedestrain
        label[label==12]=19     # bicyclist
        label[label!=19]=0
        label[label==19]=1

        edge = edge + label
        edge[edge >= 1] = 1

        return edge        



class cityscapes_video_dataset_PDA(Dataset):
    def __init__(self, data_path, gt_path, list_path):
        self.data_path = data_path
        self.gt_path = gt_path
        self.get_list(list_path)

    def __len__(self):
        return len(self.gt_label_name)

    def __getitem__(self, idx):
        img_list = []

        for name in self.img_name[idx]:
            img = cv2.imread(os.path.join(self.data_path, name))
            img = transform_im(img)
            img = torch.from_numpy(img)
            img_list.append(img)

        gt_label = cv2.imread(os.path.join(self.gt_path, self.gt_label_name[idx]), 0)

        gt_label = torch.from_numpy(gt_label.astype(np.int64))


        return img_list, gt_label

    def get_list(self, list_path):
        self.img_name = []
        self.gt_label_name = []

        with open(list_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            name, gt_label_name = line.split()
            self.gt_label_name.append(gt_label_name)
            img_id = int(name[-22:-16])
            img_name_list = []
            for j in range(10):
                img_name = name.replace('{:06d}_leftImg8bit.png'.format(img_id),
                                        '{:06d}_leftImg8bit.png'.format(img_id - 9 + j))
                img_name_list.append(img_name)
            self.img_name.append(img_name_list)

    def generate_edge(self, label, edge_width=1):
        h, w = label.shape
        edge = np.zeros(label.shape)

        # right
        edge_right = edge[1:h, :]
        edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
                & (label[:h - 1, :] != 255)] = 1

        # up
        edge_up = edge[:, :w - 1]
        edge_up[(label[:, :w - 1] != label[:, 1:w])
                & (label[:, :w - 1] != 255)
                & (label[:, 1:w] != 255)] = 1

        # upright
        edge_upright = edge[:h - 1, :w - 1]
        edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                    & (label[:h - 1, :w - 1] != 255)
                    & (label[1:h, 1:w] != 255)] = 1

        # bottomright
        edge_bottomright = edge[:h - 1, 1:w]
        edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                        & (label[:h - 1, 1:w] != 255)
                        & (label[1:h, :w - 1] != 255)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        edge = cv2.dilate(edge, kernel)

        label[label==5]=19      # pole
        label[label==11]=19      # pedestrain
        label[label==12]=19     # bicyclist
        label[label!=19]=0
        label[label==19]=1

        edge = edge + label
        edge[edge >= 1] = 1


        return edge    
