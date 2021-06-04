import h5py
from pathlib import Path

import torch
import numpy as np

from utils import gaussian_noise, random_rotation, random_translation

class Dataset(torch.utils.data.Dataset):
    '''
        A dataset class for ModelNet40
    '''
    point_clouds = np.zeros([0, 2048, 3], dtype=np.float64)
    class_ids = np.zeros([0, 1], dtype=np.uint8)

    def __init__(self, path='data', task='train', num_samples = 1024, generalize = False, robustness = False):
        self.task = task
        self.path = Path(__file__).parent.absolute() / path / 'modelnet40_ply_hdf5_2048'
        self.num_samples = num_samples
        self.generalize = generalize
        self.robustness = robustness

        if (not (self.path.exists() and self.path.is_dir())):
            print("There's No Data Directory!!")
            return

        # TODO: Paper - category split only for ply_data_trainN.h5
        filenames = "ply_data_" + (task if not generalize else '') + "*.h5"
        for filename in self.path.glob(filenames):
            with h5py.File(filename, 'r') as f:
                self.point_clouds =\
                    np.append(self.point_clouds, f['data'][:]\
                    , axis=0)
                self.class_ids =\
                    np.append(self.class_ids, f['label'][:]\
                    , axis=0)
        self.class_ids = self.class_ids.squeeze()

        if generalize:
            mask = self.class_ids < 20 # For the first 20 categories
            if task == 'train':
                self.point_clouds = self.point_clouds[mask]
                self.class_ids = self.class_ids[mask]
            else:
                self.point_clouds = self.point_clouds[1-mask]
                self.class_ids = self.class_ids[1-mask]
            

    def __getitem__(self, idx):
        # Random samples : source, target both are (N, 3)
        dev = torch.device("cpu")
        source = torch.tensor(self.point_clouds[idx][np.arange(2048)[:self.num_samples]]\
            , device=dev, dtype=torch.float32)
        #source = torch.tensor(self.point_clouds[idx][:self.num_samples]\
        #    , device=dev, dtype=torch.float32)
        target = source.clone().detach()

        # Apply gaussian noise
        # TODO: Paper - the sample noise for both
        if self.robustness:
            source += gaussian_noise()
            target += gaussian_noise()
        
        # Apply rotation
        anglexyz, rot, target = random_rotation(target, euler_order = 'XYZ')

        # Apply translation
        tr, target = random_translation(target)

        # Rotation and translation are from source to target
        return source[torch.randperm(self.num_samples)], target[torch.randperm(self.num_samples)], anglexyz, rot, tr
        #return source, target, anglexyz, rot, tr
    
    def __len__(self):
        return len(self.point_clouds)
        
# Imported from original paper Deep Closest Point
from scipy.spatial.transform import Rotation
import os
import glob

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, path='data', task='train', num_samples = 1024, generalize = False, robustness = False):
        self.data, self.label = load_data(task)
        self.num_points = num_samples
        self.partition = task
        self.gaussian_noise = robustness
        self.unseen = generalize
        self.label = self.label.squeeze()
        self.factor = 4
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.T.astype('float32'), pointcloud2.T.astype('float32'),\
               euler_ab.astype('float32'), R_ab.astype('float32'),\
               translation_ab.astype('float32')

    def __len__(self):
        return self.data.shape[0]
