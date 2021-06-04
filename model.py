import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms

import numpy as np

from utils import KNN

# PointNet was modified from Minhyuk Sung's Implementation
class PointNet(nn.Module):
    def __init__(self, in_dim=3, embed_dim=512):
        """
        PointNet: Deep Learning on Point Sets for 3D Classification and
        Segmentation.
        Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas.
        """
        super(PointNet, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(self.in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, embed_dim, 1)

        self.n1 = nn.BatchNorm1d(64)
        self.n2 = nn.BatchNorm1d(64)
        self.n3 = nn.BatchNorm1d(64)
        self.n4 = nn.BatchNorm1d(128)
        self.n5 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: (batch_size, n_classes).
        """
        x = x.transpose(2, 1)

        x = F.relu(self.n1(self.conv1(x)))
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))
        x = F.relu(self.n4(self.conv4(x)))
        x = F.relu(self.n5(self.conv5(x)))

        # x: (batch_size, embed_dim, n_points)
        return x

class DGCNN(nn.Module):
    def __init__(self, in_dim=3, embed_dim=512):
        """
            3. Linear (by Conv2D), ReLU, BatchNorm
        """
        super(DGCNN, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(self.in_dim*2, 64, 1)
        self.conv2 = nn.Conv2d(64*2, 64, 1)
        self.conv3 = nn.Conv2d(64*2, 128, 1)
        self.conv4 = nn.Conv2d(128*2, 256, 1)
        self.conv5 = nn.Conv2d(256*2, embed_dim, 1)

        self.n1 = nn.BatchNorm2d(64)
        self.n2 = nn.BatchNorm2d(64)
        self.n3 = nn.BatchNorm2d(128)
        self.n4 = nn.BatchNorm2d(256)
        self.n5 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: (batch_size, n_classes).
        """
        x = x.transpose(2, 1)
        inter = []

        x = F.relu(self.n1(self.conv1(KNN(x))))
        x = torch.max(x, -1, keepdim=True)[0]
        inter.append(x)

        x = F.relu(self.n2(self.conv2(KNN(x))))
        x = torch.max(x, -1, keepdim=True)[0]
        inter.append(x)

        x = F.relu(self.n3(self.conv3(KNN(x))))
        x = torch.max(x, -1, keepdim=True)[0]
        inter.append(x)

        x = F.relu(self.n4(self.conv4(KNN(x))))
        x = torch.max(x, -1, keepdim=True)[0]
        inter.append(x)

        x = torch.cat(inter, 1)
        # x: (batch_size, embed_dim, n_points)
        x = F.relu(self.n5(self.conv5(x))).squeeze(-1)
        return x

class Attention(nn.Module):
    '''
        Scaled Dot-Product Attention (p.3, Ch 3.2.1)
    '''
    def __init__(self, d_k):
        super(Attention, self).__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V):
        x = torch.matmul(Q, K.transpose(1, 2))/np.sqrt(self.d_k)
        return torch.matmul(torch.softmax(x, dim=2), V)

class MultiHead(nn.Module):
    '''
        Multi-Head Attention (p.4, Ch 3.2.2)
        Input : (B, N, d_model (F for previous convention))
        Output : (B, N, d_model (F for previous convention))
    '''
    def __init__(self, d_model=512, h=4):
        super(MultiHead, self).__init__()
        d_k = d_model//h
        d_v = d_model//h
        self.h = h

        self.WQ = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for i in range(h)])
        self.WK = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for i in range(h)])
        self.WV = nn.ModuleList([nn.Linear(d_model, d_v, bias=False) for i in range(h)])
        self.WO = nn.Linear(h*d_v, d_model, bias=False)
        self.head = nn.ModuleList([Attention(d_k) for i in range(h)])

    def forward(self, Q, K, V):
        x = torch.cat([self.head[i](self.WQ[i](Q), self.WK[i](K), self.WV[i](V)) for i in range(self.h)], dim=-1)
        return self.WO(x)

class FFN(nn.Module):
    '''
        Position-wise Feed-Forward Networks (p.5, Ch 3.3)
        Input : (B, N, d_model (F for previous convention))
        Output : (B, N, d_model (F for previous convention))
    '''
    def __init__(self, d_model=512, d_ff=1024):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x).transpose(1, 2))
        x = self.fc2(x.transpose(1, 2))
        return x

class Residual(nn.Module):
    '''
        Residual Connection (p.3, Ch 3.1)
    '''
    def __init__(self, d_model):
        super(Residual, self).__init__()
        self.layernorm = nn.LayerNorm([d_model])

    def forward(self, sublayer, x):
        return x + sublayer(self.layernorm(x))

class Encoder(nn.Module):
    '''
        Encoder (p.2, Ch 3.1)
        Input : (B, N, d_model (F for previous convention)) 
        Output : (B, N, d_model (F for previous convention))
    '''
    def __init__(self, d_model=512, d_ff=1024, h=4):
        super(Encoder, self).__init__()
        self.self_attention = MultiHead(d_model, h)
        self.ffn = FFN(d_model, d_ff)

        self.sublayer1 = Residual(d_model)
        self.sublayer2 = Residual(d_model)
        self.layernorm = nn.LayerNorm([d_model])
    
    def forward(self, x):
        # TODO: Test with original structure LayerNorm(x + sublayer(x)).
        # Here, supplementary material says, x + sublayer(LayerNorm(x))
        x = self.sublayer1(lambda x: self.self_attention(x, x, x), x)
        x = self.sublayer2(self.ffn, x)
        return self.layernorm(x)


class Decoder(nn.Module):
    def __init__(self, d_model=512, d_ff=1024, h=4):
        super(Decoder, self).__init__()
        self.self_attention = MultiHead(d_model, h)
        self.co_attention = MultiHead(d_model, h)
        self.ffn = FFN(d_model, d_ff)

        self.sublayer1 = Residual(d_model)
        self.sublayer2 = Residual(d_model)
        self.sublayer3 = Residual(d_model)
        self.layernorm = nn.LayerNorm([d_model])

    
    def forward(self, y, x):
        # TODO: Test with original structure LayerNorm(x + sublayer(x)).
        # Here, supplementary material says, x + sublayer(LayerNorm(x))
        x = self.sublayer1(lambda x: self.self_attention(x, x, x), x)
        x = self.sublayer2(lambda x: self.co_attention(x, y, y), x)
        x = self.sublayer3(self.ffn, x)
        return self.layernorm(x)

class Transformer(nn.Module):
    '''
        Single layer siamese transformer
          All Transformer-related functions tried to follow
        the convention of "Attention is all you need".
    '''
    def __init__(self, embed_dim=512, d_ff=1024, h=4):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embed_dim, d_ff, h)
        self.decoder = Decoder(embed_dim, d_ff, h)
    
    def forward(self, fsource, ftarget):
        fsourceT = fsource.transpose(1, 2)
        ftargetT = ftarget.transpose(1, 2)
        phisource = self.decoder(self.encoder(fsourceT), ftargetT)
        phitarget = self.decoder(self.encoder(ftargetT), fsourceT)
        return fsource + phisource.transpose(1, 2), ftarget + phitarget.transpose(1, 2)


class SVD(nn.Module):
    '''
        Mathmatical part of this implementation relied on
        https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    '''
    def __init__(self, embed_dim, device):
        super(SVD, self).__init__()
        self.embed_dim = embed_dim
        self.M = torch.eye(3).unsqueeze(0).to(device)
    
    def forward(self, source, target, fsource, ftarget):
        '''
            Input: point cloud (B, N, 3) and feature (B, F, N)
            Output: rotation (B, 3, 3) and translation (B, 3)
        '''
        # Point cloud: (B, 3, N), feature: (B, F, N)
        B = source.shape[0]
        source = source.transpose(1, 2)
        target = target.transpose(1, 2)

        # Predicted target: (B, 3, N) \hat{y}
        weight = torch.softmax(torch.matmul(ftarget.transpose(1, 2), fsource), dim=1)
        predicted_target = torch.matmul(target, weight)

        # x-\bar{x}, y-\bar{y}, (B, 3, N)
        source_mean = torch.mean(source, dim=2, keepdim=True) # (B, 3, 1)
        target_mean = torch.mean(predicted_target, dim=2, keepdim=True) # (B, 3, 1)
        source_residue = source - source_mean
        target_residue = predicted_target - target_mean

        # SVD (U, Sigma, V.T) of cross-covariance matrix S, (B, 3, 3)
        S = torch.matmul(source_residue, target_residue.transpose(1, 2))
        # U, Sigma, VT = torch.svd(S) # for 1.8.1+ (No need to transpose VT)
        UT, Sigma, V = torch.svd(S) # Here, UT is not U.T yet
        UT = UT.transpose(1, 2)

        # Orientation rectification
        M_ = self.M.expand(B, -1, -1).contiguous()
        # M_[:, 2, 2] = torch.linalg.det(torch.matmul(U, VT)) # for 1.8.1+
        M_[:, 2, 2] = torch.det(torch.matmul(V, UT))

        # Rotation and translation
        Rot = torch.matmul(torch.matmul(V, M_), UT) # (B, 3, 3)
        tr = (target_mean - torch.matmul(Rot, source_mean)).squeeze() # (B, 3, 1) -> (B, 3)
        return Rot, tr

class MLP(nn.Module):
    def __init__(self, embed_dim, device):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_dims = [2*embed_dim, 256, 128, 64]
        self.n_mlp_layers = len(self.mlp_dims) - 1
        assert(self.n_mlp_layers >= 1)

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(self.n_mlp_layers):
            self.fc.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i+1]))
            self.bn.append(nn.BatchNorm1d(self.mlp_dims[i+1]))

        self.rotation = nn.Linear(self.mlp_dims[-1], 4)
        self.translation = nn.Linear(self.mlp_dims[-1], 3)

    def forward(self, source, target, fsource, ftarget):
        '''
            Input: point cloud (B, N, 3; unused) and feature (B, F, N)
            Output: rotation (B, 3, 3) and translation (B, 3)
        '''
        x = torch.cat([fsource, ftarget], dim=1)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 2*self.embed_dim) # (B, 2F)

        for i in range(self.n_mlp_layers):
            x = F.relu(self.bn[i](self.fc[i](x)))

        # Rotation (B, 4) -> (B, 3, 3)
        rot = self.rotation(x)
        rot = rot / torch.linalg.norm(rot, dim=1, keepdim=True)
        rot = transforms.quaternion_to_matrix(rot)

        # Translation (B ,3)
        tr = self.translation(x)
        return rot, tr
