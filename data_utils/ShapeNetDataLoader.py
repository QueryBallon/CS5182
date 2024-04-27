# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
import random
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def point_cloud_noise(data, mean=0, std_dev=None):
    if std_dev is None:
        std_dev = random.uniform(0.005, 0.01)

    # noise for x, y, and z coordinates
    noise_xyz = np.random.normal(mean, std_dev, size=data[:, :3].shape)

    # noise to the original x, y, and z coordinates
    noisy_data = np.column_stack((data[:, :3] + noise_xyz, data[:, 3:]))

    return noisy_data

def point_cloud_transform(data, rotate=(0, 0, 0), scale=(1, 1, 1), shift=(0, 0, 0)):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        points = np.column_stack((x, y, z, np.ones(len(x))))

        # rotation angles (in radians)
        r_x = np.radians(rotate[0])
        r_y = np.radians(rotate[1])
        r_z = np.radians(rotate[2])

        # rotation matrices along X Y Z
        r_x_matrix = np.array([ [1, 0, 0, 0],
                                [0, np.cos(r_x), -np.sin(r_x), 0],
                                [0, np.sin(r_x), np.cos(r_x), 0],
                                [0, 0, 0, 1]])

        r_y_matrix = np.array([ [np.cos(r_y), 0, np.sin(r_y), 0],
                                [0, 1, 0, 0],
                                [-np.sin(r_y), 0, np.cos(r_y), 0],
                                [0, 0, 0, 1]])

        r_z_matrix = np.array([ [np.cos(r_z), -np.sin(r_z), 0, 0],
                                [np.sin(r_z), np.cos(r_z), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        # rotation
        combined_rotation_matrix = np.dot(r_x_matrix, np.dot(r_y_matrix, r_z_matrix))
        rotated_points = np.dot(points, combined_rotation_matrix.T)

        # scaling
        scaled_points = rotated_points * np.array([scale[0], scale[1], scale[2], 1])

        # transition
        shifted_points = scaled_points + np.array([shift[0], shift[1], shift[2], 0])

        # cartesian coordinates
        fx = shifted_points[:, 0] / shifted_points[:, 3]
        fy = shifted_points[:, 1] / shifted_points[:, 3]
        fz = shifted_points[:, 2] / shifted_points[:, 3]

        return np.column_stack((fx, fy, fz, data[:, 3:]))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        #
        self.datapath = sorted(self.datapath, key=lambda x: (x[0] != 'Chair', x[0] == 'Chair' and '7b1ca7d234f56e8a9b2cde3f4a5b6789' not in x[1]))

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            if index == 0:
                print(fn[1])
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else: 
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        # point_set = point_cloud_noise(point_set)

        point_set = point_cloud_transform(point_set,
            (
                random.randint(0, 360),
                random.randint(0, 360),
                random.randint(0, 360)
            ),
            (1, 1, 1),
            (0, 0, 0)
        )

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        choice = np.arange(self.npoints)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)



