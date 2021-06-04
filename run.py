# Minhyuk Sung (mhsung@kaist.ac.kr)

from model import DGCNN, PointNet, MLP, SVD, Transformer
from data import Dataset, ModelNet40

import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scipy.spatial.transform import Rotation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--n_epochs', type=int, default=250,
                    help='number of epochs')
parser.add_argument('--n_workers', type=int, default=4,
                    help='number of data loading workers')

parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
parser.add_argument('--step_size', nargs='+', type=int, default=[75, 150, 200], help='step size')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

parser.add_argument('--in_data_dir', type=str,
                    default='data',
                    help="data directory")
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--out_dir', type=str, default='outputs',
                    help='output directory')
parser.add_argument('--frontend', type=str, default='DGCNN', help='frontend')
parser.add_argument('--backend', type=str, default='SVD', help='backend')
parser.add_argument('--attention', action='store_true')
parser.add_argument('--embed_dims', type=int, default=512,
                    help='dimension of embeddings')
parser.add_argument('--generalize', action='store_true')
parser.add_argument('--robustness', action='store_true')
args = parser.parse_args()


# Load the data and create dataloaders.
def create_datasets_and_dataloaders(num_points=1024):
    assert(os.path.isdir(args.in_data_dir))
    #train_data = Dataset(path=args.in_data_dir, task='train',\
    #    num_samples=num_points, generalize=args.generalize, robustness=args.robustness)
    #test_data = Dataset(path=args.in_data_dir, task='test',\
    #    num_samples=num_points, generalize=args.generalize, robustness=args.robustness)
    train_data = ModelNet40(path=args.in_data_dir, task='train',\
        num_samples=num_points, generalize=args.generalize, robustness=args.robustness)
    test_data = ModelNet40(path=args.in_data_dir, task='test',\
        num_samples=num_points, generalize=args.generalize, robustness=args.robustness)

    print('# points: {:d}'.format(num_points))

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=int(args.n_workers))

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=int(args.n_workers))

    return train_data, train_dataloader, test_data, test_dataloader


# Define the mse loss function.
def compute_loss(gt_rot, gt_tr, pred_rot, pred_tr):
    # rot: (batch_size, 3, 3)
    # tr: (batch_size, 3)
    eye = torch.eye(3, device=device).unsqueeze(0).expand(gt_rot.size(0), -1, -1)
    # TODO: Where's L2 regularization?
    loss = F.mse_loss(torch.matmul(gt_rot.transpose(1, 2), pred_rot), eye) + F.mse_loss(gt_tr, pred_tr)
    return loss

# Define one-step training function.
def run_train(data, net, optimizer, writer=None):
    # Parse data.
    source, target, angles, gt_rot, gt_tr = data
    source = source.to(device)
    target = target.to(device)
    angles = angles.to(device)
    gt_rot = gt_rot.to(device)
    gt_tr = gt_tr.to(device)

    # points: (batch_size, n_points, dim_input)
    # rot: (batch_size, 3, 3)
    # tr: (batch_size, 3)
    # angles: (batch_size, 3)

    # Reset gradients.
    # https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html#zero-the-gradients-while-training-the-network
    optimizer.zero_grad()

    # Feature extraction (batch_size, embed_dim, n_points)
    fsource = net['frontend'].train()(source)
    ftarget = net['frontend'].train()(target)

    # Attention or Identity (batch_size, embed_dim, n_points)
    if args.attention:
        fsource, ftarget = net['middle'].train()(fsource, ftarget)

    # Predict transformation (batch_size, 3, 3)
    pred_rot, pred_tr = net['backend'].train()(source, target, fsource, ftarget)
    pred_angles = Rotation.from_matrix(pred_rot.detach().cpu().numpy()).as_euler('zyx')

    # Compute the loss.
    loss = compute_loss(gt_rot, gt_tr, pred_rot, pred_tr)

    # Backprop.
    loss.backward()
    optimizer.step()

    return loss, gt_rot, gt_tr, angles, pred_rot, pred_tr, pred_angles


# Define one-step evaluation function.
def run_eval(data, net, optimizer, writer=None):
    # Parse data.
    source, target, angles, gt_rot, gt_tr = data
    source = source.to(device)
    target = target.to(device)
    angles = angles.to(device)
    gt_rot = gt_rot.to(device)
    gt_tr = gt_tr.to(device)

    # points: (batch_size, n_points, dim_input)
    # rot: (batch_size, 3, 3)
    # tr: (batch_size, 3)
    # angles: (batch_size, 3)

    with torch.no_grad():
        # Feature extraction (batch_size, embed_dim, n_points)
        fsource = net['frontend'].eval()(source)
        ftarget = net['frontend'].eval()(target)

        # Attention or Identity (batch_size, embed_dim, n_points)
        if args.attention:
            fsource, ftarget = net['middle'].eval()(fsource, ftarget)

        # Predict transformation (batch_size, 3, 3)
        pred_rot, pred_tr = net['backend'].eval()(source, target, fsource, ftarget)
        pred_angles = Rotation.from_matrix(pred_rot.detach().cpu().numpy()).as_euler('zyx')

        # Compute the loss.
        loss = compute_loss(gt_rot, gt_tr, pred_rot, pred_tr)

    return loss, gt_rot, gt_tr, angles, pred_rot, pred_tr, pred_angles


# Define one-epoch training/evaluation function.
def run_epoch(dataset, dataloader, train, epoch=None, writer=None):
    total_loss = 0.0

    rot_sum_abs_error = np.zeros([0, 3])
    rot_sum_sqr_error = np.zeros([0, 3])
    tr_sum_abs_error = np.zeros([0, 3])
    tr_sum_sqr_error = np.zeros([0, 3])
    n_data = len(dataset)

    # Create a progress bar.
    pbar = tqdm(total=n_data, leave=False)

    mode = 'Train' if train else 'Test'
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)

    for i, data in enumerate(dataloader):
        torch.cuda.empty_cache()
        # Run one step.
        loss, gt_rot, gt_tr, angles, pred_rot, pred_tr, pred_angles\
            = run_train(data, net, optimizer, writer) if train else \
                run_eval(data, net, optimizer, writer)

        if train and writer is not None:
            # Write results if training.
            assert(epoch is not None)
            step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/Train', loss, step)

        batch_size = list(data[0].size())[0]
        total_loss += (loss * batch_size)

        rot_residual = np.degrees(pred_angles - angles.detach().cpu().numpy())
        tr_residual = gt_tr.detach().cpu().numpy() - pred_tr.detach().cpu().numpy()

        rot_ae = np.abs(rot_residual)
        rot_se = (rot_residual)**2
        tr_ae = np.abs(tr_residual)
        tr_se = (tr_residual)**2

        rot_sum_abs_error = np.append(rot_sum_abs_error, rot_ae, 0)
        rot_sum_sqr_error = np.append(rot_sum_sqr_error, rot_se, 0)
        tr_sum_abs_error = np.append(tr_sum_abs_error, tr_ae, 0)
        tr_sum_sqr_error = np.append(tr_sum_sqr_error, tr_se, 0)

        pbar.set_description('{} {} Loss: {:f}'.format(
            epoch_str, mode, loss))
        pbar.update(batch_size)

    pbar.close()
    mean_loss = total_loss / float(n_data)

    rot_MSE = np.mean(rot_sum_sqr_error)
    rot_MAE = np.mean(rot_sum_abs_error)
    rot_RMSE = np.sqrt(rot_MSE)
    tr_MSE = np.mean(tr_sum_sqr_error)
    tr_MAE = np.mean(tr_sum_abs_error)
    tr_RMSE = np.sqrt(tr_MSE)

    return mean_loss,\
        {'rot_MSE': rot_MSE, 'rot_RMSE': rot_RMSE, 'rot_MAE': rot_MAE,
        'tr_MSE': tr_MSE, 'tr_RMSE': tr_RMSE, 'tr_MAE': tr_MAE}


# Define one-epoch function for both training and evaluation.
def run_epoch_train_and_test(
    train_dataset, train_dataloader, test_dataset, test_dataloader, epoch=None,
        writer=None):
    train_loss, train_index = run_epoch(
        train_dataset, train_dataloader, train=args.train, epoch=epoch,
        writer=writer)
    test_loss, test_index = run_epoch(
        test_dataset, test_dataloader, train=False, epoch=epoch, writer=None)

    if writer is not None:
        # Write test results.
        assert(epoch is not None)
        step = (epoch + 1) * len(train_dataloader)
        writer.add_scalar('Loss/Test', test_loss, step)

        writer.add_scalar('rot_MSE/Train', train_index['rot_MSE'], step)
        writer.add_scalar('rot_RMSE/Train', train_index['rot_RMSE'], step)
        writer.add_scalar('rot_MAE/Train', train_index['rot_MAE'], step)
        writer.add_scalar('tr_MSE/Train', train_index['tr_MSE'], step)
        writer.add_scalar('tr_RMSE/Train', train_index['tr_RMSE'], step)
        writer.add_scalar('tr_MAE/Train', train_index['tr_MAE'], step)

        writer.add_scalar('rot_MSE/Test', test_index['rot_MSE'], step)
        writer.add_scalar('rot_RMSE/Test', test_index['rot_RMSE'], step)
        writer.add_scalar('rot_MAE/Test', test_index['rot_MAE'], step)
        writer.add_scalar('tr_MSE/Test', test_index['tr_MSE'], step)
        writer.add_scalar('tr_RMSE/Test', test_index['tr_RMSE'], step)
        writer.add_scalar('tr_MAE/Test', test_index['tr_MAE'], step)

    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)

    log = epoch_str + '\n'
    log += 'Train\nLoss: {:f}, '.format(train_loss)
    log += 'rot_MSE: {:f}, '.format(train_index['rot_MSE'])
    log += 'rot_RMSE: {:f}, '.format(train_index['rot_RMSE'])
    log += 'rot_MAE: {:f}, '.format(train_index['rot_MAE'])
    log += 'tr_MSE: {:f}, '.format(train_index['tr_MSE'])
    log += 'tr_RMSE: {:f}, '.format(train_index['tr_RMSE'])
    log += 'tr_MAE: {:f}'.format(train_index['tr_MAE'])
    log += '\n'
    log += 'Test\nLoss: {:f}, '.format(test_loss)
    log += 'rot_MSE: {:f}, '.format(test_index['rot_MSE'])
    log += 'rot_RMSE: {:f}, '.format(test_index['rot_RMSE'])
    log += 'rot_MAE: {:f}, '.format(test_index['rot_MAE'])
    log += 'tr_MSE: {:f}, '.format(test_index['tr_MSE'])
    log += 'tr_RMSE: {:f}, '.format(test_index['tr_RMSE'])
    log += 'tr_MAE: {:f}'.format(test_index['tr_MAE'])
    log += '\n\n'

    print(log)
    return test_loss


# Main function.
if __name__ == "__main__":
    print(args)

    seed=1234
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    num_points = 1024
    # Load datasets.
    train_dataset, train_dataloader, test_dataset, test_dataloader, \
         = create_datasets_and_dataloaders(num_points)

    # Create the network.
    n_dims = 3
    net = {}
    if args.frontend == 'DGCNN':
        net['frontend'] = DGCNN(n_dims, args.embed_dims)
    else:
        net['frontend'] = PointNet(n_dims, args.embed_dims)
    
    if args.attention:
        net['middle'] = Transformer(args.embed_dims)

    if args.backend == 'SVD':
        net['backend'] = SVD(args.embed_dims, device)
    else:
        net['backend'] = MLP(args.embed_dims, device)

    if torch.cuda.is_available():
        for key in net.keys():
            net[key].cuda()

    # Load a model if given.
    if args.model != '':
        checkpt = torch.load(args.model)
        for key in net.keys():
            net[key].load_state_dict(checkpt[key])

    # Set an optimizer and a scheduler.
    params = []
    for key in net.keys():
        params += list(net[key].parameters())
    optimizer = torch.optim.Adam(
        params, lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.step_size, gamma=args.gamma)

    # Create the output directory.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Train.
    if args.train:
        writer = SummaryWriter(args.out_dir)
        lowestloss = np.inf

        for epoch in range(args.n_epochs):
            loss = run_epoch_train_and_test(
                train_dataset, train_dataloader, test_dataset, test_dataloader,
                epoch, writer)

            if (epoch + 1) % 10 == 0:
                # Save the model.
                model_file = os.path.join(
                    args.out_dir, 'origin_dcp_model_{:d}.pth'.format(epoch + 1))
                torch.save({key:net[key].state_dict() for key in net.keys()}, model_file)
                print("Saved '{}'.".format(model_file))
            
            if lowestloss > loss:
                # Save the best model.
                lowestloss = loss
                model_file = os.path.join(
                    args.out_dir, 'best_dcp_model.pth')
                torch.save({key:net[key].state_dict() for key in net.keys()}, model_file)
                print("Saved '{}'.".format(model_file))

            scheduler.step()

        writer.close()
    else:
        run_epoch_train_and_test(
            train_dataset, train_dataloader, test_dataset, test_dataloader)