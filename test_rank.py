import mdp.offroad_grid as offroad_grid
import loader.data_loader as data_loader
from torch.utils.data import DataLoader
import numpy as np
import visdom

from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated
from network.reward_net import RewardNet

from torch.autograd import Variable
import torch
import time
from multiprocessing import Pool
import os
from maxent_nonlinear_offroad_rank import visualize_batch

# initialize param
grid_size = 80
discount = 0.9
batch_size = 16
n_worker = 8
#exp = '6.24'
#resume = 'step700-loss0.6980162681374217.pth'
#net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

exp = 'epoch16-11.27-rank'
resume = 'step20-nll1.0594899355382525-loss0.7099503874778748-total1.7694404125213623.pth'
net = RewardNet(n_channels=5, n_classes=1)

def rl_rank(future_traj_sample, past_traj_sample, r_sample, model, grid_size):
    svf_demo_sample = model.find_demo_svf(future_traj_sample)
    values_sample = model.find_optimal_value(r_sample, 0.01)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf(future_traj_sample, policy)
    svf_diff_sample = svf_demo_sample - svf_sample
    # (1, n_feature, grid_size, grid_size)
    svf_diff_sample = svf_diff_sample.reshape(1, 1, grid_size, grid_size)
    svf_diff_var_sample = Variable(torch.from_numpy(svf_diff_sample).float(), requires_grad=False)
    nll_sample = model.compute_nll(policy, future_traj_sample)
    dist_sample = model.compute_hausdorff_loss(policy, future_traj_sample, n_samples=1000)
    past_return_sample = model.compute_return(r_sample, past_traj_sample) # compute return
    past_return_sample = np.array([past_return_sample])
    past_return_var_sample = Variable(torch.from_numpy(past_return_sample).float())
    return nll_sample, svf_diff_var_sample, values_sample, dist_sample, past_return_var_sample


def pred_rank(feat, robot_state_feat, future_traj, past_traj, net, n_states, model, grid_size):
    n_sample = feat.shape[0]
    feat = feat.float()
    feat_var = Variable(feat)
    robot_state_feat = robot_state_feat.float()
    robot_state_feat_var = Variable(robot_state_feat)
    r_var = net(feat_var, robot_state_feat_var)

    result = []
    pool = Pool(processes=n_sample)
    for i in range(n_sample):
        r_sample = r_var[i].data.numpy().squeeze().reshape(n_states)
        future_traj_sample = future_traj[i].numpy()  # choose one sample from the batch
        future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
        future_traj_sample = future_traj_sample.astype(np.int64)
        past_traj_sample = past_traj[i].numpy()  # choose one sample from the batch
        past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
        past_traj_sample = past_traj_sample.astype(np.int64)
        result.append(pool.apply_async(rl_rank, args=(future_traj_sample, past_traj_sample, r_sample, model, grid_size)))
    pool.close()
    pool.join()
    # extract result and stack svf_diff
    nll_list = [result[i].get()[0] for i in range(n_sample)]
    dist_list = [result[i].get()[3] for i in range(n_sample)]
    svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
    values_list = [result[i].get()[2] for i in range(n_sample)]
    past_return_var_list = [result[i].get()[4] for i in range(n_sample)]
    svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
    past_return_var = torch.cat(past_return_var_list, dim=0)
    return nll_list, r_var, svf_diff_var, values_list, dist_list, past_return_var


vis = visdom.Visdom(env='test-{}'.format(exp))
model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = data_loader.OffroadLoader(grid_size=grid_size, train=False)
loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)

#net.init_weights()
checkpoint = torch.load(os.path.join('exp', exp, resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()

test_nll_list = []
test_dist_list = []
correct = 0
total = 0
for step, (feat, past_traj, future_traj, robot_state_feat, ave_energy_cons) in enumerate(loader):
    start = time.time()
    nll_list, r_var, svf_diff_var, values_list, dist_list, past_return_var = pred_rank(feat, robot_state_feat, future_traj, past_traj, net, n_states, model, grid_size)
    test_nll_list += nll_list
    test_dist_list += dist_list

    half_batch_size = past_return_var.shape[0] // 2
    past_return_var_i = past_return_var[:half_batch_size]
    past_return_var_j = past_return_var[half_batch_size:half_batch_size*2]
    output = torch.cat((past_return_var_i.unsqueeze(dim=1), past_return_var_j.unsqueeze(dim=1)), dim=1)
    ave_energy_cons_i = ave_energy_cons[:half_batch_size]
    ave_energy_cons_j = ave_energy_cons[half_batch_size:half_batch_size*2]
    target = torch.gt(ave_energy_cons_i, ave_energy_cons_j).squeeze().long() # 0 when i is better, 1 when j is better
    _, predicted = torch.max(output.data, dim=1)
    total += target.size(0)
    correct += (predicted == target).sum().item()

    #visualize_batch(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=False)
    print('{}'.format(sum(test_dist_list) / len(test_dist_list)))
nll = sum(test_nll_list) / len(test_nll_list)
dist = sum(test_dist_list) / len(test_dist_list)
accuracy = 100 * correct / total
vis.text('nll {}'.format(nll))
vis.text('distance {}'.format(dist))
vis.text('accuracy {}'.format(accuracy))
