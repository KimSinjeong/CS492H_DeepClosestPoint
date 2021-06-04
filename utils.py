import torch
from pytorch3d import transforms
import numpy as np

default_dev = torch.device("cpu")

def KNN(inp, k = 20):
    """
        Input Tensor : (B, F, N)
        1. Does KNN and return (B, F, N, K) features
        2. Concat to (B, 2F, N, K) -> Broadcasted Input tensor to (B, 2F, N, K) for concat
    """
    B, F, N = inp.shape[:3]
    inp = inp.view(B, F, N)
    nns = inp.transpose(1, 2) # nns : (B, N, F) TODO: Should this transpose be contiguous?

    # Doing KNN (B, F, N) -> (B, F, K, N)
    d = torch.cdist(nns, nns, p = 2.0, compute_mode = 'use_mm_for_euclid_dist') # d: (B, N, N)
    _, idx = torch.topk(input = d, k = k, dim = -1, largest = False) # idx : (B, N, K)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, F) # idx : (B, N, K, F)

    nns = nns.unsqueeze(2).expand(-1, -1, k, -1) # nns : (B, N, K, F)
    nns = torch.gather(nns, 1, idx) # nns : (B, N, K, F)

    return torch.cat([nns.permute(0, 3, 1, 2), inp.unsqueeze(-1).repeat(1, 1, 1, k)], 1) # (B, 2F, N, K)
    # print(torch.cuda.memory_summary(torch.device("cuda:0")))

def gaussian_noise(num_pt = 1024, std = 0.01, thr = 0.05, dev = default_dev):
    '''
        Return gaussian noise for point cloud
    '''
    return torch.clamp(torch.normal(0., std, size = (num_pt, 3), device=dev), -thr, thr)

def random_rotation(pc, theta = 45., euler_order = 'XYZ', dev = default_dev):
    '''
        Apply random rotation to point cloud and return both
        Output: Angles - radian, Rot - (3, 3) matrix, PC - Point cloud
    '''
    angles = torch.rand(size = (3, ), device=dev)*np.radians(theta)
    rot = transforms.euler_angles_to_matrix(angles, euler_order)
    return angles, rot, pc@rot.T # x.T * R.T <=> R * x

def random_translation(pc, dev = default_dev):
    '''
        Apply random translation to point cloud and return both
    '''
    tr = torch.rand(size = (3, ), device=dev) - 1.
    return tr, pc + tr.unsqueeze(0)