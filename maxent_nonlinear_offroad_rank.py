import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import warnings

warnings.filterwarnings('ignore')
from torch.autograd import Variable
import torch
from multiprocessing import Pool
from viz import overlay, feat2rgb


def overlay_traj_to_map(traj1, traj2, feat, value1=5.0, value2=10.0):
    overlay_map = feat.copy()
    for i, p in enumerate(traj1):
        overlay_map[int(p[0]), int(p[1])] = value1
    for i, p in enumerate(traj2):
        overlay_map[int(p[0]), int(p[1])] = value2
    return overlay_map


def visualize_batch(past_traj, future_traj, feat, r_var, values, svf_diff_var, optimal_traj_list, step, vis, grid_size, train=True):
    mode = 'train' if train else 'test'
    n_batch = past_traj.shape[0]
    step = step*n_batch
    for i in range(n_batch):
        future_traj_sample = future_traj[i].numpy()  # choose one sample from the batch
        future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
        future_traj_sample = future_traj_sample.astype(np.int64)
        past_traj_sample = past_traj[i].numpy()  # choose one sample from the batch
        past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
        past_traj_sample = past_traj_sample.astype(np.int64)
        optimal_traj_sample = optimal_traj_list[i]  # choose one sample from the batch
        
        overlay_map = feat[i, 0, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
        past = np.min(feat[i, 0, :, :].numpy())
        future = np.max(feat[i, 0, :, :].numpy())
        overlay_map = overlay_traj_to_map(past_traj_sample, optimal_traj_sample, overlay_map, past, future)
        vis.heatmap(X=np.flip(overlay_map, 0), opts=dict(colormap='Electric', title='{}, step {}, height'.format(mode, step+i)))
        
        # save to file
        np.savetxt('height_map/height_map_{step:02d}.txt'.format(step=step+i), X=feat[i, 0, :, :], fmt='%.2f')
        with open('optimal_traj/optimal_traj_{step:02d}.txt'.format(step=step+i), 'w') as f:
            for index in optimal_traj_sample:
                f.writelines('{} {}\n'.format(index[0], index[1]))
        
        overlay_map = overlay(feat2rgb(feat[i].numpy()), future_traj_sample, past_traj_sample)
        vis.image(np.transpose(overlay_map, (2, 0, 1)), opts=dict(title='{} rgb'.format(step+i)))
        vis.heatmap(X=np.flip(r_var.data[i].view(grid_size, -1), 0),
                    opts=dict(colormap='Electric', title='{}, step {}, rewards'.format(mode, step+i)))
        vis.heatmap(X=np.flip(svf_diff_var.data[i].view(grid_size, -1), 0),
                    opts=dict(colormap='Electric', title='{}, step {}, SVF_diff'.format(mode, step+i)))


def visualize(past_traj, future_traj, feat, r_var, values, svf_diff_var, step, vis, grid_size, train=True):
    mode = 'train' if train else 'test'

    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)
    past_traj_sample = past_traj[0].numpy()  # choose one sample from the batch
    past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
    past_traj_sample = past_traj_sample.astype(np.int64)

    vis.heatmap(X=feat[0, 0, :, :].float().view(grid_size, -1),
                opts=dict(colormap='Electric', title='{}, step {} height max'.format(mode, step)))

    overlay_map = feat[0, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
    overlay_map = overlay_traj_to_map(past_traj_sample, future_traj_sample, overlay_map)
    vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='{}, step {} height var'.format(mode, step)))

    vis.heatmap(X=feat[0, 3, :, :].float().view(grid_size, -1),
                opts=dict(colormap='Electric', title='{}, step {} green'.format(mode, step)))
    vis.heatmap(X=r_var.data[0].view(grid_size, -1),
                opts=dict(colormap='Greys', title='{}, step {}, rewards'.format(mode, step)))
    vis.heatmap(X=values[0].reshape(grid_size, -1),
                opts=dict(colormap='Greys', title='{}, step {}, value'.format(mode, step)))
    vis.heatmap(X=svf_diff_var.data[0].view(grid_size, -1),
                opts=dict(colormap='Greys', title='{}, step {}, SVF_diff'.format(mode, step)))

    # for name, param in net.named_parameters():
    #    if name.endswith('weight'):
    #        vis.histogram(param.data.view(-1), opts=dict(numbins=20))  # weights
    #        vis.histogram(param.grad.data.view(-1), opts=dict(numbins=20))  # grads


def rl(future_traj_sample, r_sample, model, grid_size):
    svf_demo_sample = model.find_demo_svf(future_traj_sample)
    values_sample = model.find_optimal_value(r_sample, 0.1)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf(future_traj_sample, policy)
    svf_diff_sample = svf_demo_sample - svf_sample
    # (1, n_feature, grid_size, grid_size)
    svf_diff_sample = svf_diff_sample.reshape(1, 1, grid_size, grid_size)
    svf_diff_var_sample = Variable(torch.from_numpy(svf_diff_sample).float(), requires_grad=False)
    nll_sample = model.compute_nll(policy, future_traj_sample)
    print(nll_sample)
    return nll_sample, svf_diff_var_sample, values_sample

def rl_rank(future_traj_sample, past_traj_sample, r_sample, model, grid_size):
    svf_demo_sample = model.find_demo_svf(future_traj_sample)
    values_sample = model.find_optimal_value(r_sample, 0.1)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf(future_traj_sample, policy)
    svf_diff_sample = svf_demo_sample - svf_sample
    # (1, n_feature, grid_size, grid_size)
    svf_diff_sample = svf_diff_sample.reshape(1, 1, grid_size, grid_size)
    svf_diff_var_sample = Variable(torch.from_numpy(svf_diff_sample).float(), requires_grad=False)
    nll_sample = model.compute_nll(policy, future_traj_sample)
    print(nll_sample)
    past_return_sample = model.compute_return(r_sample, past_traj_sample) # compute return
    past_return_sample = np.array([past_return_sample])
    past_return_var_sample = Variable(torch.from_numpy(past_return_sample).float())
    return nll_sample, svf_diff_var_sample, values_sample, past_return_var_sample

def pred(feat, robot_state_feat, future_traj, net, n_states, model, grid_size):
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
        result.append(pool.apply_async(rl, args=(future_traj_sample, r_sample, model, grid_size)))
    pool.close()
    pool.join()
    # extract result and stack svf_diff
    nll_list = [result[i].get()[0] for i in range(n_sample)]
    svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
    values_list = [result[i].get()[2] for i in range(n_sample)]
    svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
    return nll_list, r_var, svf_diff_var, values_list

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
    svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
    values_list = [result[i].get()[2] for i in range(n_sample)]
    past_return_var_list = [result[i].get()[3] for i in range(n_sample)]
    svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
    past_return_var = torch.cat(past_return_var_list, dim=0)
    return nll_list, r_var, svf_diff_var, values_list, past_return_var
