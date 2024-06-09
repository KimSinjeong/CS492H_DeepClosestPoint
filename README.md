# CS492(H) Deep Closest Point
[Deep Closest Point](https://arxiv.org/abs/1905.03304) reproduction for KAIST CS492(H) spring final project. Here, my [proposal](https://drive.google.com/file/d/1zbClvdDk8E-tqIwgB7WRauUIluBGsFUn/view?usp=sharing), [report](https://drive.google.com/file/d/1bZuJscIpN7m5CEaxohOUaz6jShQePMmk/view?usp=sharing), [supplementary material](https://drive.google.com/file/d/1b-ogQ2oNrKhde_I8R5Zx95MKWWb2L80W/view?usp=sharing), [poster](https://drive.google.com/file/d/1MEZGCOTreCAVppZqafn-mSmW-uwNJYV7/view?usp=sharing) are provided.

## Dataset
1. Download https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
2. Create a directory named 'data' under this repository and unzip the zip file under the directory.

## Pretrained model
All my pretrained models are [here](https://drive.google.com/drive/folders/1h3JO9boA9S8ZcnCQ1EPHms9q8YltPZEb?usp=drive_link).

## Requirements
I only mention here strict requirements that I found.

* PyTorch==1.7.1

* PyTorch3D

* VGA with VRAM larger than or equal to 24GB

## How to train
```bash
python run.py --train --attention // Default model (DCP-v2)
python run.py --train // DCP-v1
python run.py --train --attention --frontend=PointNet // Use PointNet for frontend instead of DGCNN (Ablation study 1)
python run.py --train --attention --backend=MLP // Use multi-layer perceptron for backend instead of SVD (Ablation study 2)
python run.py --train --attention --unseen // Train and Test for different categories
python run.py --train --attention --embed_dims=<DIMENSION of FEATURE VECTOR> // Set dimension of feature vector different from 512. (Set 768 to get the results for ablation study 3)
```

## How to test
```bash
// You can use the same options with train
python run.py --model=<MODEL PATH> --attention
```

## Acknowledgements
1. For fair comparison, data loader was borrowed from original work(https://github.com/WangYueFt/dcp) with some modification.
2. Training code (run.py) and PointNet are borrowed and modified from Minhyuk Sung's skeleton code for KAIST CS492(H) homework 1.
