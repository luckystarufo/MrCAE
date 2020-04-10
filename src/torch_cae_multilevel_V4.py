import os
import torch
import pickle
import numpy as np
from utils import MultiScaleDynamicsDataSet, \
    apply_local_op, check_pixel_level_loss, \
    apply_mask, compute_loss_all_scales


class Conv2dBlock(torch.nn.Module):
    def __init__(self, num_of_blocks=1, num_of_channels=1, mode='conv', is_widen=True,
                 activation=torch.nn.ReLU(), std=0.02):
        super(Conv2dBlock, self).__init__()
        assert isinstance(num_of_blocks, int) and num_of_blocks >= 1, \
            print('num_of_blocks must be a positive integer!')
        # params
        self.num_of_channels = num_of_channels
        self.num_of_blocks = num_of_blocks
        self.is_widen = is_widen
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # activation function
        self.activation = activation

        # convolution / deconvolution layers
        if mode == 'conv':
            self.add_module('B0', torch.nn.Conv2d(1, num_of_channels, 3, stride=2, padding=0).to(self.device))
            for i in range(1, num_of_blocks):
                self.add_module('B{}'.format(i), torch.nn.Conv2d(num_of_channels, num_of_channels, 3, stride=1, padding=1).
                                to(self.device))
        elif mode == 'deconv':
            for i in range(num_of_blocks - 1):
                self.add_module('B{}'.format(i), torch.nn.ConvTranspose2d(num_of_channels, num_of_channels, 3,
                                                                          stride=1, padding=1).to(self.device))
            self.add_module('B{}'.format(num_of_blocks - 1), torch.nn.ConvTranspose2d(num_of_channels,
                                                                                      1, 3, stride=2,
                                                                                      padding=0).to(self.device))
        else:
            raise ValueError('mode can be conv or deconv ONLY!')

        # init
        if is_widen:
            self.apply_rand_init(std)
        else:
            self.apply_identity_init(std)

    def forward(self, x):
        h = x
        for i in range(self.num_of_blocks):
            h = self._modules['B{}'.format(i)](h)
            if self.is_widen and self.mode == 'conv':
                h = self.activation(h)

        return h

    def apply_identity_init(self, std=0.02):
        m = 1.0 / self.num_of_channels
        if self.mode == 'conv':
            self._modules['B0'].weight.data.uniform_(std, std)
            self._modules['B0'].weight.data[0, 0, 1, 1] += 1.0
        elif self.mode == 'deconv':
            self._modules['B0'].weight.data.uniform_(-std, +std)
            self._modules['B0'].weight.data[:, :, 1, 1] += m
        self._modules['B0'].bias.data.uniform_(-std, std)

        for i in range(1, self.num_of_blocks - 1):
            self._modules['B{}'.format(i)].weight.data.uniform_(-std, +std)
            self._modules['B{}'.format(i)].weight.data[:, :, 1, 1] += m
            self._modules['B{}'.format(i)].bias.data.uniform_(-std, std)

        if self.mode == 'conv':
            if self.num_of_blocks > 1:
                self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data.uniform_(-std, +std)
                self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 1, 1] += m
        elif self.mode == 'deconv':
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data.uniform_(m/4 - std, m/4 + std)
            # bi-linear interpolation?
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 0, 1] += m / 4
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 1, 0] += m / 4
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 1, 2] += m / 4
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 2, 1] += m / 4
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 1, 1] += m / 4
            self._modules['B{}'.format(self.num_of_blocks - 1)].weight.data[:, :, 1, 1] += m / 2

        self._modules['B{}'.format(self.num_of_blocks - 1)].bias.data.uniform_(-std, std)

    def apply_rand_init(self, std=0.02):
        for i in range(self.num_of_blocks):
            self._modules['B{}'.format(i)].weight.data.uniform_(-std, std)
            self._modules['B{}'.format(i)].bias.data.uniform_(-std, std)


class CAE(torch.nn.Module):
    def __init__(self, n_levels, n_blocks=1, activation=torch.nn.ReLU(), use_maps=False):
        """
        :param n_levels: maximum level of the network
        :param n_blocks: how many convolution/deconvolution layers for each approximation block? (depth)
        :param activation: activation function
        :param use_maps: if to use maps to give feedback and do progressive refinement
        return None
        """
        super(CAE, self).__init__()

        # pre-process
        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * n_levels
        assert isinstance(n_blocks, list)

        # param
        self.cur_level = -1
        self.n_levels = n_levels
        self.blocks = n_blocks
        self.use_maps = use_maps

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # training info
        self.level_clear = dict()
        self.filter_channels_each_level = dict()
        self.n_filters_each_level = dict()
        self.n_filter_groups_each_level = dict()
        self.resolved_maps = dict()
        self.n_params = list()

        for i in range(n_levels):
            self.n_filters_each_level[str(i)] = 0
            self.filter_channels_each_level[str(i)] = []
            self.n_filter_groups_each_level[str(i)] = 0
            self.resolved_maps[str(i)] = dict()
            self.level_clear[str(i)] = False

        # layer
        self.activation = activation

    def forward(self, x, level, query_in_out_each_level=False, query_hidden=False):
        """
        :param x: a 4D input of NN
        :param level: level index
        :param query_in_out_each_level: if to query the input/output at each level
        (maybe used for enforce losses at different levels)
        :param query_hidden: if to query hidden representations
        :return: output of NN, a list of hidden representations at current level
        """
        # collectors
        all_hidden = dict()
        all_inputs = dict()
        all_outputs = dict()

        # forward prop
        assert level >= 0, print('level index should be a non-negative integer!')
        resolved_maps_dict = self.resolved_maps[str(level)]
        if level == 0:
            if query_in_out_each_level:
                all_inputs['0'] = x
            encoded = self._modules['L0_Conv_0'](x)
            if query_hidden:
                all_hidden['L0_0'] = encoded
            # ----- pad -----
            encoded = torch.nn.functional.pad(encoded, (1, 1, 1, 1), 'replicate')
            # ---------------
            y = self._modules['L0_deConv_0'](encoded)
            # chop off the boundaries
            y = y[:, :, 2:-2, 2:-2]
            for i in range(1, self.n_filter_groups_each_level['0']):
                encoded = self._modules['L0_Conv_{}'.format(i)](x)
                if self.use_maps:
                    masked_encoded = apply_mask(encoded, resolved_maps_dict[str(i - 1)])
                else:
                    masked_encoded = encoded
                if query_hidden:
                    all_hidden['L0_{}'.format(i)] = masked_encoded
                y += self._modules['L0_deConv_{}'.format(i)](masked_encoded)
            if query_in_out_each_level:
                all_outputs['0'] = y
        else:
            encoded = self._modules['L{}_Conv_0'.format(level)](x)
            decoded, ins, outs, hs = \
                self.forward(encoded, level-1, query_in_out_each_level, query_hidden)
            # ----- pad -----
            decoded = torch.nn.functional.pad(decoded, (1, 1, 1, 1), 'replicate')
            # ---------------
            if query_in_out_each_level:
                all_inputs[str(level)] = x
                all_inputs.update(ins)
                all_outputs.update(outs)
            if query_hidden:
                all_hidden.update(hs)
                all_hidden['L{}_0'.format(level)] = encoded
            y = self._modules['L{}_deConv_0'.format(level)](decoded)
            y = y[:, :, 2:-2, 2:-2]
            for i in range(1, self.n_filter_groups_each_level[str(level)]):
                encoded = self._modules['L{}_Conv_{}'.format(level, i)](x)
                if self.use_maps:
                    masked_encoded = apply_mask(encoded, resolved_maps_dict[str(i - 1)])
                else:
                    masked_encoded = encoded
                if query_hidden:
                    all_hidden['L{}_{}'.format(level, i)] = masked_encoded
                y += self._modules['L{}_deConv_{}'.format(level, i)](masked_encoded)
            if query_in_out_each_level:
                all_outputs[str(level)] = y

        return y, all_inputs, all_outputs, all_hidden

    def deeper_op(self, std=0.02):
        """
        perform a deepening operation such that we inherit what we have learned in
        previous level, and we offer a higher level resolution data with a layer
        attaching to it.
        :param std: standard deviation from the center
        :return: None
        """
        if self.cur_level + 1 >= self.n_levels:
            print('the network has reached to its deepest level! Abort.')
        else:
            # level increase
            self.cur_level += 1
            # add two new layers
            n_blocks = self.blocks[self.cur_level]
            # the first layer at each level doesn't need nonlinearity
            self.add_module('L{}_Conv_{}'.format(self.cur_level, 0),
                            Conv2dBlock(n_blocks, 1, mode='conv', is_widen=False, std=std))
            self.add_module('L{}_deConv_{}'.format(self.cur_level, 0),
                            Conv2dBlock(1, 1, mode='deconv', is_widen=False, std=std))
            # update info
            self.n_filters_each_level[str(self.cur_level)] += 1
            self.filter_channels_each_level[str(self.cur_level)].append(1)
            self.n_filter_groups_each_level[str(self.cur_level)] += 1
        self.n_params.append(sum(p.numel() for p in self.parameters()))

    def wider_op(self, n_filters, std=0.02):
        """
        perform a widening operation such that we expand the capacity of the model at
        current level
        :param n_filters: number of filters to add
        :param std: standard deviation for init
        :return: None
        """
        if self.level_clear[str(self.cur_level)]:
            print('this level is clear, no need to widen the network! Abort.')
        else:
            # add two layers
            n_blocks = self.blocks[self.cur_level]
            filter_index = self.n_filter_groups_each_level[str(self.cur_level)]
            self.add_module('L{}_Conv_{}'.format(self.cur_level, filter_index),
                            Conv2dBlock(n_blocks, n_filters, mode='conv', activation=self.activation,
                                        is_widen=True, std=std))
            self.add_module('L{}_deConv_{}'.format(self.cur_level, filter_index),
                            Conv2dBlock(1, n_filters, mode='deconv', activation=self.activation,
                                        is_widen=True, std=std))
            # update info
            self.n_filters_each_level[str(self.cur_level)] += n_filters
            self.filter_channels_each_level[str(self.cur_level)].append(n_filters)
            self.n_filter_groups_each_level[str(self.cur_level)] += 1
        self.n_params.append(sum(p.numel() for p in self.parameters()))

    def train_on_this_level_wrapper(self, dataset, max_epoch, batch_size, widen_sizes,
                                    tol=None, lr=1e-3, std=0.02, verbose=1, w=0.5, model_path=None):
        """
        :param dataset: a MultiScaleDynamicsDataSet object
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param widen_sizes: a list of integers specifying widening size at each step
        :param tol: a float, represent error tolerance or set to None as default
        :param lr: learning rate
        :param std: noise level to break the symmetry for init
        :param verbose: verbose level, controls the print-out message
        :param w: loss = w * l2_loss + (1-w) * l_inf_loss
        :param model_path: path that is used to save the model at each level
        :return: three lists of training info
        """
        # pre-process
        if isinstance(widen_sizes, int):
            widen_sizes = [widen_sizes]
        assert isinstance(widen_sizes, list), \
            print('widen sizes should be a list of positive integers!')

        # collectors
        arch = list()
        n_params = list()
        all_val_losses = list()
        best_val_losses = list()
        max_pos_set = set()

        print('*************************************************')
        print('Model @Level {}:'.format(self.cur_level+1))
        print('Perform deepening & widening, train each architectures ...')

        # deepen
        self.deeper_op(std=std)
        if verbose > 1:
            print('model layers: ')
            print(list(self._modules.keys()))
        val_losses, best_val_loss, mset = self.train_arch(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr, tol=tol, verbose=verbose,w=w)
        # collect results & save model
        arch.append(1)
        max_pos_set.update(mset)
        n_params.append(sum(p.numel() for p in self.parameters()))
        all_val_losses.append(val_losses)
        best_val_losses.append(best_val_loss)

        filter_index = self.n_filter_groups_each_level[str(self.cur_level)] - 1
        if model_path is not None:
            torch.save(self, os.path.join(model_path, 'model_L{}_{}.pt'.
                                          format(self.cur_level, filter_index)))
        # log
        print('')
        if verbose > 1 and tol is not None:
            print('level {}, resolved map {} (after training):'.format(self.cur_level, filter_index))
            print(self.resolved_maps[str(self.cur_level)][str(filter_index)])

        print('-------------------------------------------------')

        # widen
        cnt = 0
        while not self.level_clear[str(self.cur_level)] and cnt < len(widen_sizes):
            n_filters = widen_sizes[cnt]
            cnt += 1
            if verbose > 1:
                print('prepare attaching {} more filters to current level arch ...'.format(n_filters))
            self.wider_op(n_filters=n_filters, std=std)
            if verbose > 1:
                print('model layers: ')
                print(list(self._modules.keys()))
            val_losses, best_val_loss, mset = self.train_arch(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr, tol=tol, verbose=verbose, w=w)
            # collect results & save model
            arch.append(arch[-1] + n_filters)
            max_pos_set.update(mset)
            n_params.append(sum(p.numel() for p in self.parameters()))
            all_val_losses.append(val_losses)
            best_val_losses.append(best_val_loss)
            filter_index = self.n_filter_groups_each_level[str(self.cur_level)] - 1
            if model_path is not None:
                torch.save(self, os.path.join(model_path, 'model_L{}_{}.pt'.
                                              format(self.cur_level, filter_index)))
            # log
            print('')
            if verbose > 1 and tol is not None:
                print('level {}, resolved map {} (after training):'.format(self.cur_level, filter_index))
                print(self.resolved_maps[str(self.cur_level)][str(filter_index)])

            print('-------------------------------------------------')

        print('*************************************************')

        return arch, n_params, all_val_losses, best_val_losses, max_pos_set

    def train_arch(self, dataset, max_epoch, batch_size,
                   tol=None, lr=1e-3, w=0.5, verbose=1):
        """
        :param dataset: a MultiScaleDynamicsDataSet object
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param tol: error tolerance (default is None)
        :param lr: learning rate
        :param w: w: loss = w * l2_loss + (1-w) * l_inf_loss
        :param verbose: verbose level
        :return: a list of train_losses, val_losses and timings
        """
        # prepare data at this level
        train_data, val_data, _ = dataset.obtain_data_at_current_level(self.cur_level)

        # specify optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, weight_decay=1e-5)
        criterion = torch.nn.MSELoss(reduction='none')

        # collectors
        val_losses = list()
        train_losses = list()
        max_pos_set = set()

        # training
        epoch = 0
        ave_loss_old = 1e+10
        best_local_val_err = 1e+10
        best_state_dict = self.state_dict()
        while epoch < max_epoch:
            epoch += 1
            # =================== forward =====================
            new_idxs = torch.randperm(dataset.n_train)
            batch_idxs = new_idxs[:batch_size]
            batch_train_data = train_data[batch_idxs, :, :, :]
            output, _, _, _ = self.forward(batch_train_data, self.cur_level)
            output_val, _, _, _ = self.forward(val_data, self.cur_level)
            # =============== calculate losses ================
            mean_loss_train = criterion(output, batch_train_data).mean()
            max_loss_train = criterion(output, batch_train_data).mean(0).max()
            assert 0 <= w <= 1, print('w should between 0 and 1 (inclusive)!')
            loss = w * mean_loss_train + (1 - w) * max_loss_train
            mean_loss_val = criterion(output_val, val_data).mean()
            max_loss_val = criterion(output_val, val_data).mean(0).max()
            loss_val = w * mean_loss_val + (1 - w) * max_loss_val
            if loss_val.item() < best_local_val_err:
                best_local_val_err = loss_val.item()
                best_state_dict = self.state_dict()
            # compute global scale losses
            global_mean_loss, global_max_loss, _ = self.compute_global_loss(dataset, output, self.cur_level, dataset.train_inds[batch_idxs])
            global_mean_val_loss, global_max_val_loss, tup = self.compute_global_loss(dataset, output_val, self.cur_level, dataset.val_inds)
            global_loss = w * global_mean_loss + (1-w) * global_max_loss
            global_val_loss = w * global_mean_val_loss + (1-w) * global_max_val_loss
            # =================== backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ================= collect stat ==================
            train_losses.append(global_loss.item())
            val_losses.append(global_val_loss.item())
            max_pos_set.add(tup)
            # =================== log =========================
            # 1st epoch
            if epoch == 1 and verbose:
                # init err
                print('losses printing format: local: mse/max/overall, global: mse/max/overall')
                print('epoch [1/{}]'.format(max_epoch))
                print('[training set] local: {:.4f}/{:.4f}/{:.4f}, global: {:.4f}/{:.4f}/{:.4f}'.format(mean_loss_train.item(), max_loss_train.item(), loss.item(), global_mean_loss.item(), global_max_loss.item(), global_loss.item()))
                print('[validation set] local: {:.4f}/{:.4f}/{:.4f}, global: {:.4f}/{:.4f}/{:.4f}'.format(mean_loss_val.item(), max_loss_val.item(), loss_val.item(), global_mean_val_loss.item(), global_max_val_loss.item(), global_val_loss.item()))
            # every 1/10 max_epoch
            if epoch % (max_epoch // 10) == 0:
                print('epoch [{}/{}]:'.format(epoch, max_epoch))
                print('[training set] local: {:.4f}/{:.4f}/{:.4f}, global: {:.4f}/{:.4f}/{:.4f}'.format(mean_loss_train.item(), max_loss_train.item(), loss.item(), global_mean_loss.item(), global_max_loss.item(), global_loss.item()))
                print('[validation set] local: {:.4f}/{:.4f}/{:.4f}, global: {:.4f}/{:.4f}/{:.4f}'.format(mean_loss_val.item(), max_loss_val.item(), loss_val.item(), global_mean_val_loss.item(), global_max_val_loss.item(), global_val_loss.item()))
                # check for early stopping
                if tol is not None:
                    # check if fully resolved?
                    train_output, _, _, _ = self.forward(train_data, self.cur_level)
                    fully_resolved, _, _ = check_pixel_level_loss(train_data, train_output, tol=tol, device=self.device, w=0.5)
                    if fully_resolved and epoch < max_epoch:
                        print('early stopping at {}th iteration due to satisfying reconstruction!'.format(epoch))
                        break
                if epoch > max_epoch // 2 and epoch < max_epoch:
                    ave_loss = np.mean(train_losses[-(max_epoch // 10):])
                    if (ave_loss_old - ave_loss) / ave_loss_old < 1e-3:
                        # improvement is so small that we consider it as convergence
                        print('early stopping at {}th iteration due to slow convergence!'.format(epoch))
                        break
                    ave_loss_old = ave_loss

        # calculate the best validation error
        self.load_state_dict(best_state_dict)
        output_val, _, _, _ = self.forward(val_data, self.cur_level)
        global_mean_val_loss, global_max_val_loss, _ = self.compute_global_loss(dataset, output_val, self.cur_level, dataset.val_inds)
        global_val_loss = w * global_mean_val_loss + (1-w) * global_max_val_loss
        best_val_loss = global_val_loss.item()
        # check this level is clear if tolerance threshold is enabled
        filter_index = self.n_filter_groups_each_level[str(self.cur_level)] - 1
        train_output, _, _, _ = self.forward(train_data, self.cur_level)
        if tol is not None:
            _, _, resolved_map = check_pixel_level_loss(train_data, train_output, tol=tol, device=self.device, w=0.5)
            self.resolved_maps[str(self.cur_level)][str(filter_index)] = resolved_map.float()
            if resolved_map.all():
                self.level_clear[str(self.cur_level)] = True

        return val_losses, best_val_loss, max_pos_set

    def compute_global_loss(self, dataset, output, cur_level, inds):
        """
        although the loss function is defined independently on each level, we
        provide this function to track the global training progress
        :param dataset: a MultiScaleDynamicsDataSet object
        :param output: the output at this level
        :param cur_level: current level
        :param inds: indicies
        :return: loss at the global scale
        """
        # obtain data
        data = dataset.data[inds].to(dataset.device)
        n_data = data.size()[0]

        # up-sample the small one
        n = self.n_levels - cur_level - 1
        for _ in range(n):
            tmp = torch.nn.functional.pad(output, (1, 1, 1, 1), 'replicate')
            output = apply_local_op(tmp, self.device, 'deconv')
            output = output[:, :, 2:-2, 2:-2]

        # calculate the loss
        criterion = torch.nn.MSELoss(reduction='none')
        mse_loss = criterion(output, data).mean()
        max_loss = criterion(output*dataset.map_data, data*dataset.map_data).mean(0).max()
        max_pos = criterion(output*dataset.map_data, data*dataset.map_data).mean(0).argmax()
        #
        x = max_pos.item() % output.size()[-1]
        y = max_pos.item() // output.size()[-1]
        tup = (x, y)

        return mse_loss, max_loss, tup


def train_net(archs, dataset, max_epoch, batch_size, result_path,
              tols=None, model_path=None, activation=torch.nn.ReLU(),
              lr=1e-3, w=0.5, std=0.02, verbose=1):
    """
    :param archs: a list of lists, specify the architectures
    :param tols: a list or None, specify the tolerance of each level (with respect to the global error)
    :param activation: activation function of the archs
    :param dataset: a MultiScaleDynamicsDataSet object
    :param max_epoch: maximum number of epochs
    :param batch_size: batch size
    :param lr: learning rate
    :param w: loss = w * l2_loss + (1-w) * l_inf_loss
    :param std: standard deviation for init
    :param model_path: path to save the models
    :param result_path: path to save the results
    :param verbose: verbose level
    :return: a dictionary with the following key-value pairs:
            - 1. 'model': model object (final)
            - 2. 'n_params': a list of number of parameters at different stages
            - 3. 'n_encodings': a list of message sizes at different stages
            - 4. 'best_val_errs': a list of validation errors of the best model at
                                  the end of each stage
            - 5. 'full_val_errs': a list of lists of validation errors throughout
                                  the training of each stage
    """
    # preprocess
    n_levels = dataset.n_levels
    assert n_levels == len(archs), print('levels of dataset and architecture are not consistent!')
    if tols is None:
        tols = [None] * n_levels
        use_maps = False
    else:
        use_maps = True

    # create model
    n_levels = dataset.n_levels
    model = CAE(n_levels=n_levels, activation=activation, use_maps=use_maps)

    # training
    for i in range(n_levels):
        model = train_net_one_level(arch=archs[i], dataset=dataset, max_epoch=max_epoch,
                                    batch_size=batch_size, result_path=result_path,
                                    model_path=model_path, load_model=None, tol=tols[i],
                                    activation=activation, lr=lr, w=w, std=std,
                                    verbose=verbose)

    return model


def train_net_one_level(arch, dataset, max_epoch, batch_size, result_path,
                        model_path=None, load_model=None, tol=None,
                        activation=torch.nn.ReLU(), lr=1e-3, w=0.5,
                        std=0.02, verbose=1):
    """
    :param arch: a list of lists, specify the architecture of the networks
    :param tol: specify the tolerance of this level (with respect to the global error)
    :param activation: activation function of the archs
    :param dataset: a MultiScaleDynamicsDataSet object
    :param max_epoch: maximum number of epochs
    :param batch_size: batch size
    :param load_model: path to the model to be loaded (train the next level)
    :param lr: learning rate
    :param w: loss = w * l2_loss + (1-w) * l_inf_loss
    :param std: standard deviation for init
    :param model_path: path to save the models
    :param result_path: path to save the results
    :param verbose: verbose level
    :return: a dictionary with the following key-value pairs:
            - 1. 'model': model object (final)
            - 2. 'n_params': a list of number of parameters at different stages
            - 3. 'n_encodings': a list of message sizes at different stages
            - 4. 'best_val_errs': a list of validation errors of the best model at
                                   the end of each stage
            - 5. 'full_val_errs': a list of lists of validation errors throughout
                                    the training of each stage
    """
    # preprocess
    if tol is not None:
        assert isinstance(tol, float), print('tols should be a float.')
        use_maps = True
    else:
        use_maps = False

    # create/load the model
    n_levels = dataset.n_levels
    if load_model is None:
        model = CAE(n_levels=n_levels, activation=activation, use_maps=use_maps)
    else:
        model = load_model
    widen_sizes = [arch[k+1] - arch[k] for k in range(len(arch)-1)]

    # training
    # perform a deepening operation
    model = train_net_one_stage(mode=1, n_filters=1, dataset=dataset, max_epoch=max_epoch,
                                batch_size=batch_size, result_path=result_path, tol=tol,
                                load_model=model, activation=activation, lr=lr, w=w, std=std,
                                model_path=model_path, verbose=verbose)
    # perform a sequence of widening operations
    cnt = 0
    while not model.level_clear[str(model.cur_level)] and cnt < len(widen_sizes):
        n_filters = widen_sizes[cnt]
        cnt += 1
        model = train_net_one_stage(mode=0, n_filters=n_filters, dataset=dataset, max_epoch=max_epoch,
                                    batch_size=batch_size, result_path=result_path, tol=tol,
                                    load_model=model, activation=activation, lr=lr, w=w, std=std,
                                    model_path=model_path, verbose=verbose)

    return model


def train_net_one_stage(mode, n_filters, dataset, max_epoch, batch_size, result_path,
                        load_model=None, tol=None, activation=torch.nn.ReLU(), lr=1e-3,
                        w=0.5, std=0.02, model_path=None, verbose=1):
    """
    :param mode: 1 or 0, 1 represents deepen op and 0 represents widen op
    :param n_filters: number of filters to allocate
    :param tol: a list or None, specify the tolerance of each level (with respect to the global error)
    :param activation: activation function of the archs
    :param dataset: a MultiScaleDynamicsDataSet object
    :param max_epoch: maximum number of epochs
    :param batch_size: batch size
    :param lr: learning rate
    :param w: loss = w * l2_loss + (1-w) * l_inf_loss
    :param std: standard deviation for init
    :param model_path: path to save the models
    :param result_path: path to save the results
    :param load_model: path to the model to be loaded (train the next level)
    :param verbose: verbose level
    :return: new model
    """
    # collectors
    if load_model is None or result_path is None:
        final_arch = list()
        n_params = list()
        full_val_errs = list()
        best_val_errs = list()
        max_pos_set = set()
    else:
        with open(os.path.join(result_path, 'val_errs.dat'), "rb") as f:
            full_val_errs = pickle.load(f)
        with open(os.path.join(result_path, 'best_errs.dat'), "rb") as f:
            best_val_errs = pickle.load(f)
        with open(os.path.join(result_path, 'n_params.dat'), "rb") as f:
            n_params = pickle.load(f)
        with open(os.path.join(result_path, 'arch.dat'), "rb") as f:
            final_arch = pickle.load(f)
        with open(os.path.join(result_path, 'max_pos_set.dat'), "rb") as f:
            max_pos_set = pickle.load(f)

    # preprocess
    assert (mode == 0 or mode == 1), print("mode: invalid input!")
    if mode:
        n_filters = 1
    if tol is not None:
        assert isinstance(tol, float), print('tols should be a float.')
        use_maps = True
    else:
        use_maps = False

    # create/load the model
    n_levels = dataset.n_levels
    if load_model is None:
        model = CAE(n_levels=n_levels, activation=activation, use_maps=use_maps)
    else:
        model = load_model

    # training
    if mode:
        # deepening operation
        print('*************************************************')
        print('Model @Level {}:'.format(model.cur_level + 1))
        print('Perform deepening & widening, train each architectures ...')
        model.deeper_op(std=std)
        if verbose > 1:
            print('model layers: ')
            print(list(model._modules.keys()))
        val_losses, best_val_loss, mset = model.train_arch(dataset, max_epoch=max_epoch, batch_size=batch_size,
                                                           lr=lr, tol=tol, verbose=verbose, w=w)
        # collect results & save model
        final_arch.append([1])
        max_pos_set.update(mset)
        n_params.append(sum(p.numel() for p in model.parameters()))
        best_val_errs.append(best_val_loss)
        full_val_errs.append([val_losses])
        filter_index = model.n_filter_groups_each_level[str(model.cur_level)] - 1
        if model_path is not None:
            torch.save(model, os.path.join(model_path, 'model_L{}_{}.pt'.format(model.cur_level, filter_index)))
        print('-------------------------------------------------')
    else:
        # widening operation
        if verbose > 1:
            print('prepare attaching {} more filters to current level arch ...'.format(n_filters))
        model.wider_op(n_filters=n_filters, std=std)
        if verbose > 1:
            print('model layers: ')
            print(list(model._modules.keys()))
        val_losses, best_val_loss, mset = model.train_arch(dataset, max_epoch=max_epoch, batch_size=batch_size,
                                                           lr=lr, tol=tol, verbose=verbose, w=w)
        # collect results & save model
        final_arch[-1].append(final_arch[-1][-1] + n_filters)
        max_pos_set.update(mset)
        n_params.append(sum(p.numel() for p in model.parameters()))
        best_val_errs.append(best_val_loss)
        full_val_errs[-1].append(val_losses)
        filter_index = model.n_filter_groups_each_level[str(model.cur_level)] - 1
        if model_path is not None:
            torch.save(model, os.path.join(model_path, 'model_L{}_{}.pt'.format(model.cur_level, filter_index)))
        # log
        print('')
        if verbose > 1 and tol is not None:
            print('level {}, resolved map {} (after training):'.format(model.cur_level, filter_index))
            print(model.resolved_maps[str(model.cur_level)][str(filter_index)])
        print('-------------------------------------------------')

    # save the results
    with open(os.path.join(result_path, 'val_errs.dat'), "wb") as fp:
        pickle.dump(full_val_errs, fp)
    with open(os.path.join(result_path, 'best_errs.dat'), "wb") as fp:
        pickle.dump(best_val_errs, fp)
    with open(os.path.join(result_path, 'n_params.dat'), "wb") as fp:
        pickle.dump(n_params, fp)
    with open(os.path.join(result_path, 'arch.dat'), "wb") as fp:
        pickle.dump(final_arch, fp)
    with open(os.path.join(result_path, 'max_pos_set.dat'), "wb") as fp:
        pickle.dump(max_pos_set, fp)

    return model
