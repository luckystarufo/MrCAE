import numpy as np
import torch


class MultiScaleDynamicsDataSet():
    def __init__(self, data_path, n_levels, map_path=None, train_ratio=0.7, valid_ratio=0.2):
        # load data
        data = np.load(data_path)
        self.data = torch.tensor(data).unsqueeze(1).float()
        #
        if map_path is not None:
            map_data = 1 - np.load(map_path)
            self.map_data = torch.tensor(map_data).float()
        else:
            self.map_data = torch.ones(data.shape[-2:]).float()

        self.nt, self.nx, self.ny = data.shape
        # partition
        indices = np.arange(self.nt)
        np.random.shuffle(indices)
        n_train = int(train_ratio*self.nt)
        n_val = int(valid_ratio*self.nt)
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = self.nt - n_train - n_val
        self.train_inds = indices[:n_train]
        self.val_inds = indices[n_train:n_train+n_val]
        self.test_inds = indices[n_train+n_val:]
        #
        self.n_levels = n_levels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.map_data = self.map_data.to(self.device)

    def obtain_data_at_current_level(self, level):
        train_data = self.data[self.train_inds].to(self.device)
        val_data = self.data[self.val_inds].to(self.device)
        test_data = self.data[self.test_inds].to(self.device)

        for _ in range(self.n_levels - level - 1):
            train_data = apply_local_op(train_data, self.device, ave=False)
            val_data = apply_local_op(val_data, self.device, ave=False)
            test_data = apply_local_op(test_data, self.device, ave=False)

        return train_data, val_data, test_data


def apply_local_op(data, device, mode='conv', ave=True):
    """
    :param data: data to be processed
    :param device: which device is the data placed in?
    :param mode: string, 'conv' or 'deconv'
    :param ave: if to use local average or sample the center
    :return: processed data
    """
    in_channels, out_channels, _, _ = data.size()
    n = min(in_channels, out_channels)
    if mode == 'conv':
        op = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=0).to(device)
    elif mode == 'deconv':
        op = torch.nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=0).to(device)
    else:
        raise ValueError('mode can only be conv or deconv!')
    op.weight.data = torch.zeros(op.weight.data.size()).to(device)
    op.bias.data = torch.zeros(op.bias.data.size()).to(device)

    for i in range(n):
        if mode == 'conv':
            if ave:
                op.weight.data[i, i, :, :] = torch.ones(op.weight.data[i, i, :, :].size()).to(device) / 9
            else:
                op.weight.data[i, i, 1, 1] = torch.ones(op.weight.data[i, i, 1, 1].size()).to(device)
        elif mode == 'deconv':
            op.weight.data[i, i, :, :] = torch.ones(op.weight.data[i, i, :, :].size()).to(device) / 4
            op.weight.data[i, i, 0, 1] += 1 / 4
            op.weight.data[i, i, 1, 0] += 1 / 4
            op.weight.data[i, i, 1, 2] += 1 / 4
            op.weight.data[i, i, 2, 1] += 1 / 4
            op.weight.data[i, i, 1, 1] += 1 / 4
            op.weight.data[i, i, 1, 1] += 1 / 2

    # make them non-trainable
    for param in op.parameters():
        param.requires_grad = False

    return op(data)


def check_pixel_level_loss(d1, d2, tol, device, w=0.5):
    """
    :param d1: data 1
    :param d2: data 2
    :param tol: a float, represent the tolerance
    :param device: device
    :param w: loss = w * mse_loss + (1 - w) * max_loss
    :return: a boolean value, if error satisfies the tolerance,
             a torch tensor of overall loss distribution,
             and a boolean torch tensor
    """
    assert isinstance(tol, float), print('tol should be a float!')

    loss1 = torch.mean((d1 - d2)**2, dim=0, keepdim=True)
    loss2 = torch.max((d1 - d2)**2, dim=0, keepdim=True)[0]
    loss = w * loss1 + (1 - w) * loss2
    loss_summary = apply_local_op(loss, device).squeeze()

    return loss_summary.max() <= tol, loss_summary, loss_summary <= tol


def apply_mask(data, mask, mask_type='resolved', width=1):
    """
    :param data: data to be processed
    :param mask: mask, a 2D torch tensor of 0s and 1s
    :param mask_type: resolved map or unresolved map
    :param width: int, specify how large the region is
    :return: a 4D torch tensor represents masked data
    """
    if not isinstance(width, int):
        raise ValueError('width should be a positive integer!')

    # convert to unresolved mask
    if mask_type == 'resolved':
        mask = 1 - mask
    elif mask_type == 'unresolved':
        mask = mask
    else:
        raise ValueError('mask_type could only be resolved or unresolved!')

    # expansion
    dx = [i for i in range(-width, width+1)]
    dy = [i for i in range(-width, width+1)]
    m, n = mask.size()
    for c in mask.nonzero():
        x, y = int(c[0]), int(c[1])
        for i in range(2*width+1):
            for j in range(2*width+1):
                if 0 <= x + dx[i] < m and 0 <= y + dy[j] < n:
                    mask[x + dx[i], y + dy[j]] = 1

    # apply
    masked_data = data * mask.unsqueeze(0).unsqueeze(0).float()
    return masked_data


def compute_loss_all_scales(in_dict, out_dict):
    """
    :param in_dict: a dict which contains inputs at all scales
    :param out_dict: a dict which contains outputs at all scales
    :return: MSE loss at all scales
    """
    loss = 0.0
    assert set(in_dict.keys()) == set(out_dict.keys()), \
        print('inputs and outputs are not consistent!')
    criterion = torch.nn.MSELoss()
    for k in in_dict.keys():
        loss += criterion(in_dict[k], out_dict[k])

    return loss
