import os
import torch
import pickle
import numpy as np
from utils import MultiScaleDynamicsDataSet, \
    apply_local_op, check_pixel_level_loss, \
    apply_mask, compute_loss_all_scales


class CAE(torch.nn.Module):
    def __init__(self, n_levels, channels, activation=torch.nn.ReLU(), std=0.02):
        """
        :param n_levels: number of levels
        :param channels: number of channels of each level
        :param activation: activation function
        :param std: standard deviation for init
        """
        super(CAE, self).__init__()
        self.n_levels = n_levels
        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert n_levels == len(channels) - 1, \
            print("inputs are not consistent!")

        for i in range(n_levels):
            self.add_module('E{}'.format(i+1),
                            torch.nn.Conv2d(channels[i], channels[i+1],
                                            kernel_size=3, stride=2).to(self.device))
            self._modules['E{}'.format(i+1)].weight.data.uniform_(-std, std)
            self._modules['E{}'.format(i+1)].bias.data.uniform_(-std, std)

        for i in range(n_levels):
            self.add_module('D{}'.format(n_levels - i),
                            torch.nn.ConvTranspose2d(channels[-i-1], channels[-i-2],
                                                     kernel_size=3, stride=2).to(self.device))
            self._modules['D{}'.format(n_levels - i)].weight.data.uniform_(-std, std)
            self._modules['D{}'.format(n_levels - i)].bias.data.uniform_(-std, std)

    def forward(self, x):
        """
        :param x: input
        :return: reconstruction
        """
        hs = list()
        hs.append(x)
        h = x
        # encoding
        for i in range(self.n_levels):
            h = self.activation(self._modules['E{}'.format(i+1)](h))
            hs.append(h)

        # decoding
        x_hat = hs[-1]
        for i in range(self.n_levels - 1):
            x_hat = self._modules['D{}'.format(self.n_levels - i)](x_hat) + hs[-i-2]
        x_hat = self._modules['D1'](x_hat)

        return x_hat

    def train(self, dataset, level_diff, max_epoch, batch_size, lr=1e-3, w=0.5):
        """
        :param dataset: a MultiScaleDynamicsDataSet object
        :param level_diff: difference in level
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param lr: learning rate
        :param w: w: loss = w * l2_loss + (1-w) * l_inf_loss
        :return: the best performance on the validation set (globally)
        """
        # prepare data at this level
        train_data, val_data, _ = dataset.obtain_data_at_current_level(dataset.n_levels - level_diff - 1)
        # specify optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, weight_decay=1e-5)
        criterion = torch.nn.MSELoss(reduction='none')

        # run epochs
        epoch = 0
        ave_loss_old = 1e+10
        best_val_err = 1e+10
        train_losses = list()
        best_state_dict = self.state_dict()
        while epoch < max_epoch:
            epoch += 1
            epoch += 1
            # =================== forward =====================
            new_idxs = torch.randperm(dataset.n_train)
            batch_idxs = new_idxs[:batch_size]
            batch_train_data = train_data[batch_idxs, :, :, :]
            output = self.forward(batch_train_data)
            output_val = self.forward(val_data)
            # =============== calculate losses ================
            mean_loss_train = criterion(output, batch_train_data).mean()
            max_loss_train = criterion(output, batch_train_data).mean(0).max()
            assert 0 <= w <= 1, print('w should between 0 and 1 (inclusive)!')
            loss = w * mean_loss_train + (1 - w) * max_loss_train
            mean_loss_val = criterion(output_val, val_data).mean()
            max_loss_val = criterion(output_val, val_data).mean(0).max()
            loss_val = w * mean_loss_val + (1 - w) * max_loss_val
            if loss_val.item() < best_val_err:
                best_val_err = loss_val.item()
                best_state_dict = self.state_dict()
            # compute global scale losses
            global_mean_loss, global_max_loss = self.compute_global_loss(dataset, output, level_diff, dataset.train_inds[batch_idxs])
            global_mean_val_loss, global_max_val_loss = self.compute_global_loss(dataset, output_val, level_diff, dataset.val_inds)
            global_loss = w * global_mean_loss + (1-w) * global_max_loss
            global_val_loss = w * global_mean_val_loss + (1-w) * global_max_val_loss
            train_losses.append(global_loss.item())
            # =================== backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ====================== log ======================
            if epoch == 1:
                print('losses printing format: local: mse/max/overall, global: mse')
            if epoch % (max_epoch // 10) == 0:
                print('epoch [{}/{}]:'.format(epoch, max_epoch))
                print('[training set] local: {:.4f}/{:.4f}/{:.4f}, global: {:.4f}/{:.4f}/{:.4f}'.format(mean_loss_train.item(), max_loss_train.item(), loss.item(), global_mean_loss.item(), global_max_loss.item(), global_loss.item()))
                print('[validation set] local: {:.4f}/{:.4f}/{:.4f}, global: {:.4f}/{:.4f}/{:.4f}'.format(mean_loss_val.item(), max_loss_val.item(), loss_val.item(), global_mean_val_loss.item(), global_max_val_loss.item(), global_val_loss.item()))
                # check for early stopping
                if epoch > max_epoch // 2:
                    ave_loss = np.mean(train_losses[-(max_epoch // 10):])
                    if (ave_loss_old - ave_loss) / ave_loss_old < 1e-3:
                        # improvement is so small that we consider it as convergence
                        print('early stopping at {}th iteration due to slow convergence!'.format(epoch))
                        break
                    ave_loss_old = ave_loss

        # store the best error on validation set
        self.load_state_dict(best_state_dict)
        output_val = self.forward(val_data)
        global_mean_val_loss, global_max_val_loss = self.compute_global_loss(dataset, output_val, level_diff, dataset.val_inds)
        global_val_loss = w * global_mean_val_loss + (1-w) * global_max_val_loss
        err = global_val_loss.item()

        return err

    def compute_global_loss(self, dataset, output, level_diff, inds):
        """
        although the loss function is defined independently on each level, we
        provide this function to track the global training progress :)
        :param dataset: a MultiScaleDynamicsDataSet object
        :param output: the output at this level
        :param level_diff: current level
        :param inds: indicies
        :return: loss at the global scale
        """
        # obtain data
        data = dataset.data[inds].to(dataset.device)
        n_data = data.size()[0]

        # up-sample the small one
        for _ in range(level_diff):
            tmp = torch.nn.functional.pad(output, (1, 1, 1, 1), 'replicate')
            output = apply_local_op(tmp, self.device, 'deconv')
            output = output[:, :, 2:-2, 2:-2]

        # calculate the loss
        criterion = torch.nn.MSELoss(reduction='none')
        mse_loss = criterion(output, data).mean()
        max_loss = criterion(output, data).mean(0).max()

        return mse_loss, max_loss


def train_archs(arch, activation, dataset, base_epoch, batch_size,
                result_path, model_path=None, resume_training=False,
                lr=1e-3, w=0.5, std=0.02):
    """
    :param arch: a list of lists, specify the architecture of the networks
    :param activation: activation function of the archs
    :param dataset: a MultiScaleDynamicsDataSet object
    :param base_epoch: base number of epochs
    :param batch_size: batch size
    :param result_path: path to save the results
    :param model_path: path to save the models
    :param resume_training: if to resume training or train from the beginning
    :param lr: learning rate
    :param w: loss = w * l2_loss + (1-w) * l_inf_loss
    :param std: standard deviation for init
    :return: None
    """
    if resume_training:
        with open(os.path.join(result_path, 'n_params_REDNet.dat'), "rb") as f:
            n_params = pickle.load(f)
        with open(os.path.join(result_path, 'best_errs_REDNet.dat'), "rb") as f:
            errs = pickle.load(f)
    else:
        n_params = list()
        errs = list()

    n_levels = len(arch)
    nt = 0
    for k in range(n_levels):
        for j in range(len(arch[k])):
            nt += 1
            max_epoch = nt * base_epoch
            level_diff = dataset.n_levels - k - 1
            channels = list((1, arch[k][j]))
            channels += [arch[i][-1] for i in range(k - 1, -1, -1)]
            net = CAE(n_levels=k+1, channels=channels, activation=activation, std=std)
            n = sum(p.numel() for p in net.parameters())
            if len(n_params) == 0 or n > n_params[-1]:
                err = net.train(dataset, level_diff, max_epoch, batch_size, lr, w)
                print("-------------------------------------------------")
                n_params.append(n)
                errs.append(err)
