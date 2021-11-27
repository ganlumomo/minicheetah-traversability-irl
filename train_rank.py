import mdp.offroad_grid as offroad_grid
import loader.data_loader as data_loader
from torch.utils.data import DataLoader
import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import visdom
import warnings
import logging
import os

warnings.filterwarnings('ignore')
from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated
from network.one_stage_dilated import OneStageDilated
from network.reward_net import RewardNet
from torch.autograd import Variable
import torch
import time
from maxent_nonlinear_offroad_rank import pred, pred_rank, rl, overlay_traj_to_map, visualize

logging.basicConfig(filename='maxent_nonlinear_offroad.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)

""" init param """
#pre_train_weight = 'pre-train-v6-dilated/step1580-loss0.0022763446904718876.pth'
pre_train_weight = None
vis_per_steps = 20
test_per_steps = 20
# resume = 'step130-loss1.2918855668639102.pth'
resume = None
exp_name = 'epoch16-11.27-rank'
grid_size = 80
discount = 0.9
lr = 5e-4
n_epoch = 16
batch_size = 16
n_worker = 8
use_gpu = True

if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))

#host = os.environ['HOSTNAME']
#vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://128.2.176.221', port=4546)
vis = visdom.Visdom(env='main')

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

train_loader = data_loader.OffroadLoader(grid_size=grid_size, tangent=False, more_kinematic=None)
train_loader = DataLoader(train_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)
test_loader = data_loader.OffroadLoader(grid_size=grid_size, train=False, tangent=False)
test_loader = DataLoader(test_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)

net = RewardNet(n_channels=5, n_classes=1)
step = 0
nll_cma = 0
nll_test = 0
loss_cma = 0
loss_test = 0

if resume is None:
    if pre_train_weight is None:
       #net.init_weights()
       pass
    else:
        pre_train_check = torch.load(os.path.join('exp', pre_train_weight))
        net.init_with_pre_train(pre_train_check)
else:
    checkpoint = torch.load(os.path.join('exp', exp_name, resume))
    step = checkpoint['step']
    net.load_state_dict(checkpoint['net_state'])
    nll_cma = checkpoint['nll_cma']
    # opt.load_state_dict(checkpoint['opt_state'])

opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
loss_criterion = torch.nn.CrossEntropyLoss()

train_nll_win = vis.line(X=np.array([[-1, -1]]), Y=np.array([[nll_cma, nll_cma]]),
                         opts=dict(xlabel='steps', ylabel='nll', title='train acc'))
test_nll_win = vis.line(X=np.array([-1]), Y=np.array([nll_test]),
                        opts=dict(xlabel='steps', ylabel='nll', title='test acc'))

train_loss_win = vis.line(X=np.array([[-1, -1]]), Y=np.array([[loss_cma, loss_cma]]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train loss'))
test_loss_win = vis.line(X=np.array([-1]), Y=np.array([loss_test]),
                        opts=dict(xlabel='steps', ylabel='loss', title='test loss'))

""" train """
best_test_nll = np.inf
best_test_loss = np.inf
for epoch in range(n_epoch):
    for _, (feat, past_traj, future_traj, robot_state_feat, ave_energy_cons) in enumerate(train_loader):
        start = time.time()
        net.train()
        print('main. step {}'.format(step))

        nll_list, r_var, svf_diff_var, values_list, past_return_var = pred_rank(feat, robot_state_feat, future_traj, past_traj, net, n_states, model, grid_size)

        opt.zero_grad()
        # a hack to enable backprop in pytorch with a vector
        # the normally used loss.backward() only works when loss is a scalar
        torch.autograd.backward([r_var], [-svf_diff_var])  # to maximize, hence add minus sign

        # Trajectory Ranking
        half_batch_size = past_return_var.shape[0] // 2
        past_return_var_i = past_return_var[:half_batch_size]
        past_return_var_j = past_return_var[half_batch_size:half_batch_size*2]
        output = torch.cat((past_return_var_i.unsqueeze(dim=1), past_return_var_j.unsqueeze(dim=1)), dim=1)
        ave_energy_cons_i = ave_energy_cons[:half_batch_size]
        ave_energy_cons_j = ave_energy_cons[half_batch_size:half_batch_size*2]
        target = torch.gt(ave_energy_cons_i, ave_energy_cons_j).squeeze().long() # 0 when i is better, 1 when j is better
        loss = loss_criterion(output, target)
        loss_var = Variable(loss, requires_grad=True)
        loss_var.backward()

        opt.step()
        nll = sum(nll_list) / len(nll_list)
        print('main. acc {}. loss {}. took {} s'.format(nll, loss, time.time() - start))

        # cma. cumulative moving average. window size < 20
        nll_cma = (nll + nll_cma * min(step, 20)) / (min(step, 20) + 1)
        vis.line(X=np.array([[step, step]]), Y=np.array([[nll, nll_cma]]), win=train_nll_win, update='append')
        loss_cma = (loss + loss_cma * min(step, 20)) / (min(step, 20) + 1)
        vis.line(X=np.array([[step, step]]), Y=np.array([[loss, loss_cma]]), win=train_loss_win, update='append')

        if step % vis_per_steps == 0 or nll > 2.5:
            visualize(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=True)
            if step == 0:
                step += 1
                continue

        if step % test_per_steps == 0:
            # test
            net.eval()
            nll_test_list = []
            loss_test_list = []
            for _, (feat, past_traj, future_traj, robot_state_feat, ave_energy_cons) in enumerate(test_loader):
                tmp_nll, r_var, svf_diff_var, values_list, past_return_var = pred_rank(feat, robot_state_feat, future_traj, past_traj, net, n_states, model, grid_size)
                nll_test_list += tmp_nll

                half_batch_size = past_return_var.shape[0] // 2
                past_return_var_i = past_return_var[:half_batch_size]
                past_return_var_j = past_return_var[half_batch_size:half_batch_size*2]
                output = torch.cat((past_return_var_i.unsqueeze(dim=1), past_return_var_j.unsqueeze(dim=1)), dim=1)
                ave_energy_cons_i = ave_energy_cons[:half_batch_size]
                ave_energy_cons_j = ave_energy_cons[half_batch_size:half_batch_size*2]
                target = torch.gt(ave_energy_cons_i, ave_energy_cons_j).squeeze().long() # 0 when i is better, 1 when j is better
                tmp_loss = loss_criterion(output, target)
                loss_test_list.append(tmp_loss)

            nll_test = sum(nll_test_list) / len(nll_test_list)
            loss_test = sum(loss_test_list) / len(loss_test_list)
            print('main. test nll {}. test loss {}'.format(nll_test, loss_test))
            vis.line(X=np.array([step]), Y=np.array([nll_test]), win=test_nll_win, update='append')
            vis.line(X=np.array([step]), Y=np.array([loss_test]), win=test_loss_win, update='append')
            # visualize(feat, r_variable, values, svf_diff_var, step, train=False)

            # if getting best test results, save weights
            if nll_test < best_test_nll:
                best_test_nll = nll_test
                state = {'nll_cma': nll_cma, 'test_nll': nll_test, 'loss_cma': loss_cma, 'test_loss': loss_test, 'step': step, 'net_state': net.state_dict(),
                         'opt_state': opt.state_dict(), 'discount':discount}
                path = os.path.join('exp', exp_name, 'step{}-nll{}-loss{}-total{}.pth'.format(step, nll_test, loss_test, nll_test+loss_test))
                torch.save(state, path)

            if loss_test < best_test_loss:
                best_test_loss = loss_test
                state = {'nll_cma': nll_cma, 'test_nll': nll_test, 'loss_cma': loss_cma, 'test_loss': loss_test, 'step': step, 'net_state': net.state_dict(),
                         'opt_state': opt.state_dict(), 'discount':discount}
                path = os.path.join('exp', exp_name, 'step{}-nll{}-loss{}-total{}.pth'.format(step, nll_test, loss_test, nll_test+loss_test))
                torch.save(state, path)
        
        step += 1
