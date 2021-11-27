clear; clc;
close all;

% For computing running means and stds
% global elevation_meanobj uncertainty_meanobj r_meanobj g_meanobj b_meanobj
% global elevation_stdobj uncertainty_stdobj r_stdobj g_stdobj b_stdobj
% elevation_meanobj = vision.Mean('RunningMean', true);
% uncertainty_meanobj = vision.Mean('RunningMean', true);
% r_meanobj = vision.Mean('RunningMean', true);
% g_meanobj = vision.Mean('RunningMean', true);
% b_meanobj = vision.Mean('RunningMean', true);
% elevation_stdobj = vision.StandardDeviation('RunningStandardDeviation', true);
% uncertainty_stdobj = vision.StandardDeviation('RunningStandardDeviation', true);
% r_stdobj = vision.StandardDeviation('RunningStandardDeviation', true);
% g_stdobj = vision.StandardDeviation('RunningStandardDeviation', true);
% b_stdobj = vision.StandardDeviation('RunningStandardDeviation', true);

% Generate training data
grid_size = 80;
window_size = 450;
folder_path = '/home/ganlu/minicheetah_irldata/';
%folder_path = '/media/ganlu/Samsung_T5/0000_mini-cheetah/2021-05-29_Forest_Sidewalk_Rock_Data/2021-05-29-00-51-42/raw_data/'
file_list = dir(append(folder_path, '*.xml'));
for i = 300:length(file_list)
    file_path = append(folder_path, file_list(i).name);
    process_file(file_path, grid_size, window_size, true, true);
end
%save('/home/ganlu/Docker/vehicle-motion-forecasting/scripts/running_statistics.mat', 'elevation_meanobj', 'uncertainty_meanobj', ...
  %  'r_meanobj', 'g_meanobj', 'b_meanobj', 'elevation_stdobj', 'uncertainty_stdobj', 'r_stdobj', 'g_stdobj', 'b_stdobj');

function process_file(file_path, grid_size, window_size, visualize, generate)
    file = readstruct(file_path);
    elevation = read_grid(file.elevation, grid_size);
    uncertainty = read_grid(file.uncertainty, grid_size);
    r = read_grid(file.r, grid_size);
    g = read_grid(file.g, grid_size);
    b = read_grid(file.b, grid_size);
    
    past_traj = read_trajectory(file.past_traj);
    future_traj = read_trajectory(file.future_traj);
    valid_past_traj = validate_trajectory(past_traj);
    valid_future_traj = validate_trajectory(future_traj);
    valid_traj = validate_trajectory([past_traj; future_traj]);
    imu_data = process_imu(file.past_imu, window_size);
    [joint_state_data, average_energy_consumption] = process_joint_state(file.past_joint_state, window_size, valid_traj);
    robot_state_data = [imu_data joint_state_data];
    
    if visualize & min(sum(var(valid_past_traj)), sum(var(valid_future_traj))) > 10
        rgb_overlayed = overlay_traj_to_rgb(r, g, b, valid_past_traj, valid_future_traj, valid_traj);
        visualize_grid(elevation, uncertainty, rgb_overlayed, robot_state_data, average_energy_consumption);
    end
    if generate & min(sum(var(valid_past_traj)), sum(var(valid_future_traj))) > 10
        generate_mat(file_path, elevation, uncertainty, r, g, b, ...
            valid_past_traj, valid_future_traj, robot_state_data, average_energy_consumption, grid_size);
    end
end

function grid = read_grid(layer, grid_size)
    str = layer.data;
    strs = split(str);
    nums = str2double(strs);
    grid = reshape(nums, grid_size, grid_size)';
end

function traj = read_trajectory(traj_file)
    strs = split(traj_file);
    nums = str2double(strs);
    traj = reshape(nums, 2, size(nums, 1)/2)'; % 0-based indexing
end

function imu_data = process_imu(past_imu, window_size)
    strs = split(past_imu);
    nums = str2double(strs);
    imus = reshape(nums, 6, size(nums,1)/6)';
    angular_velocity = imus(end-window_size*2+2:2:end, 1:3);
    linear_acceleration = imus(end-window_size*2+2:2:end, 4:6);
    imu_data = [angular_velocity linear_acceleration];
end

function [joint_state_data, average_energy_consumption] = process_joint_state(past_joint_state, window_size, valid_traj)
    strs = split(past_joint_state);
    nums = str2double(strs);
    joint_states = reshape(nums, 3, size(nums,1)/3)';
    position = reshape(joint_states(:,1), 12, size(joint_states(:,1),1)/12)';
    velocity = reshape(joint_states(:,2), 12, size(joint_states(:,2),1)/12)';
    effort = reshape(joint_states(:,3), 12, size(joint_states(:,3),1)/12)';
    joint_state_data = [position(end-window_size+1:end,:) velocity(end-window_size+1:end,:)];
    n = size(position, 1)-1;
    A = [-eye(n),zeros(n,1)]+[zeros(n,1),eye(n)];
    position_displacement = A * position;
    average_energy_consumption = sum(abs(diag(effort(1:end-1,:)*position_displacement'))) / size(valid_traj, 1);
end

function traj_interpolated = interpolate_trajectory(traj)
    sz = size(traj, 1);
    xy = [traj(:,1)'; traj(:,2)'];
    pp = spline(1:sz, xy);
    traj_interpolated = int8(ppval(pp, linspace(1,sz))');
end

function traj_valid = validate_trajectory(traj)
    current_state = traj(1,:);
    traj_valid(1,:) = current_state;
    i = 2;
    j = 2;
    while j <= size(traj, 1)
        if sum(abs(current_state-traj(j,:))) == 0
            j = j+1;
            continue
        elseif sum(abs(current_state-traj(j,:))) > 1
            if abs(current_state(1)-traj(j,1)) >= abs(current_state(2)-traj(j,2))
                current_state(1) = current_state(1) + sign(traj(j,1)-current_state(1));
            else
                current_state(2) = current_state(2) + sign(traj(j,2)-current_state(2));
            end
            traj_valid(i,:) = current_state;
            i = i+1;
        else
            current_state = traj(j,:);
            traj_valid(i,:) = current_state;
            i = i+1;
            j = j+1;
        end
    end
%     if size(traj_valid, 1) >= grid_size
%         traj_valid = traj_valid(1:grid_size-1, :);
%     end
end

% Remove the exact successive states
function traj_filtered = filter_trajectory(traj)
    traj_filtered(1,:) = traj(1,:);
    i = 2;
    for j = 2:size(traj, 1)
        if traj(j, :) == traj_filtered(i-1, :)
            continue
        end
        traj_filtered(i, :) = traj(j, :);
        i = i+1;
    end
end

function rgb_overlayed = overlay_traj_to_rgb(r, g, b, past_traj, future_traj, traj)
    past_traj = past_traj + 1;
    future_traj = future_traj + 1;
    traj = traj + 1;
    for i = 1:size(past_traj, 1)
        r(past_traj(i, 1), past_traj(i, 2)) = 1;
        g(past_traj(i, 1), past_traj(i, 2)) = 0;
        b(past_traj(i, 1), past_traj(i, 2)) = 0;
    end
    for i = 1:size(future_traj, 1)
        r(future_traj(i, 1), future_traj(i, 2)) = 0;
        g(future_traj(i, 1), future_traj(i, 2)) = 1;
        b(future_traj(i, 1), future_traj(i, 2)) = 1;
    end
%     for i = 1:size(traj, 1)
%         r(traj(i, 1), traj(i, 2)) = 0;
%         g(traj(i, 1), traj(i, 2)) = 0;
%         b(traj(i, 1), traj(i, 2)) = 1;
%     end
    rgb_overlayed(:,:,1) = r;
    rgb_overlayed(:,:,2) = g;
    rgb_overlayed(:,:,3) = b;
end

function visualize_grid(elevation, uncertainty, rgb_overlayed, robot_state_data, average_energy_consumption)
    load('plasma.mat');
    figure;
    subplot(1,4,1);
    imagesc(elevation);
    colormap(plasma);
    subplot(1,4,2);
    imagesc(uncertainty);
    subplot(1,4,3);
    imagesc(rgb_overlayed);
    subplot(1,4,4);
    imagesc(robot_state_data);
    title(average_energy_consumption);
    plot_robot_state(robot_state_data);
end

function plot_robot_state(robot_state)
    figure;
    subplot(2,1,1);
    plot(robot_state(:,1),'LineWidth',2);
    set(gca,'FontSize', 13, 'TickLabelInterpreter','latex'); hold on
    plot(robot_state(:,2),'LineWidth',2); hold on
    plot(robot_state(:,3),'LineWidth',2);
    legend({'$\omega_x$', '$\omega_y$', '$\omega_z$'}, 'Interpreter', 'latex', 'FontSize', 17);
    subplot(2,1,2);
    plot(robot_state(:,4),'LineWidth',2);
    set(gca,'FontSize', 13, 'TickLabelInterpreter','latex'); hold on
    plot(robot_state(:,5),'LineWidth',2); hold on
    plot(robot_state(:,6),'LineWidth',2);
    legend({'$a_x$', '$a_y$', '$a_z$'}, 'Interpreter', 'latex', 'FontSize', 17);
    
end

function generate_mat(file_path, elevation, uncertainty, r, g, b, past_traj, future_traj, robot_state_data, average_energy_consumption, grid_size)
% global elevation_meanobj uncertainty_meanobj r_meanobj g_meanobj b_meanobj
% global elevation_stdobj uncertainty_stdobj r_stdobj g_stdobj b_stdobj
    feat = zeros(5, size(r,1), size(r,2));
    feat(1,:,:) = elevation;
    for i = 1:grid_size
        for j = 1:grid_size

        end
    end
    feat(2,:,:) = uncertainty;
    for i = 1:grid_size
        for j = 1:grid_size
  
        end
    end
    feat(3,:,:) = 255*r;
    feat(4,:,:) = 255*g;
    feat(5,:,:) = 255*b;
%     for i = 1:grid_size
%         for j = 1:grid_size
%             elevation_meanobj(elevation(i,j));
%             elevation_stdobj(elevation(i,j));
%             uncertainty_meanobj(uncertainty(i,j));
%             uncertainty_stdobj(uncertainty(i,j));
%             r_meanobj(255*r(i,j));
%             r_stdobj(255*r(i,j));
%             g_meanobj(255*g(i,j));
%             g_stdobj(255*g(i,j));
%             b_meanobj(255*b(i,j));
%             b_stdobj(255*b(i,j));
%         end
%     end
    save_path = append(file_path(1:end-4), '.mat');
    save(save_path, 'feat', 'robot_state_data', 'past_traj', 'future_traj', 'robot_state_data', 'average_energy_consumption');
end
