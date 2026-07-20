import os
import sys
import torch
import numpy as np
from numpy import random
from torch.utils.data import DataLoader, Dataset
# cut the observation, action, position, reward, BEV, agent, target
import math


def expend_data(
    observations,
    actions_behavior_id,
    actions_behavior_val,
    actions_label_id,
    actions_label_val,
    rewards,
    command,
    actions_behavior_prior,
    percentage=2,
):

    def expend_delta(n_start, n_end, percentage):
        fractional_part, integer_part = math.modf(percentage)
        _split = []
        for i in range(int(integer_part)):
            _split.append([n_start, n_end])
        delta = int((n_end - n_start) * fractional_part)
        _split.append([n_end - delta, n_end])
        return _split

    split_id = []  # select the start and end of the data
    n_start = 0
    n_end = 0

    flag = False  # flag为True: 下一个end没用
    for i in range(len(observations)):
        if actions_behavior_id[i] == 16:
            if flag:
                flag = False
                continue
            flag = True

            n_end = i+1
            split_id.extend(expend_delta(n_start, n_end, percentage))
            n_start = n_end

    # print("split_id",split_id)
    (
        _obs_arr,
        _bact_id_arr,
        _lact_id_arr,
        _bact_val_arr,
        _lact_val_arr,
        _reward_arr,
        _command_arr,
        _bact_prior_arr
    ) = ([], [], [], [], [], [], [], [])

    for split in split_id:
        n_b = split[0]
        n_e = split[1]

        _obs_arr.extend(observations[n_b : (n_e + 1)])
        _bact_id_arr.extend(
            actions_behavior_id[n_b : n_e + 1]
        ) 
        _bact_val_arr.extend(
            actions_behavior_val[n_b : n_e + 1]
        ) 
        _lact_id_arr.extend(actions_label_id[n_b : n_e + 1]) 
        _lact_val_arr.extend(
            actions_label_val[n_b : n_e + 1]
        ) 
        _reward_arr.extend(rewards[n_b : n_e + 1]) 
        _command_arr.extend(command[n_b : n_e + 1])
        if actions_behavior_prior is not None:
            _bact_prior_arr.extend(actions_behavior_prior[n_b : n_e + 1])
        

    return (
        np.array(_obs_arr),
        np.array(_bact_id_arr),
        np.array(_bact_val_arr),
        np.array(_lact_id_arr),
        np.array(_lact_val_arr),
        np.array(_reward_arr),
        np.array(_command_arr),
        np.array(_bact_prior_arr)
    )

def cut_data(
    observations,
    actions_behavior_id,
    actions_behavior_val,
    actions_label_id,
    actions_label_val,
    rewards,
    command,
    actions_behavior_prior,
    percentage=1,
    time_step=2_000,
):

    split_id = []  # select the start and end of the data
    n_start = 0
    n_end = 0
    n_episode = 0

    flag = False  # flag为True: 下一个end没用
    for i in range(len(observations)):
        if actions_behavior_id[i] == 16:  # end

            if flag:
                flag = False
                continue

            flag = True

            n_episode += 1
            n_end = i + 1 # 跳过全黑

            delta = n_end - n_start
            # delta = int((n_end - n_start) * percentage)
            if delta == 0:
                delta = 1
            n_start = n_end - delta
            split_id.append((n_start, n_end))
            n_start = n_end+1
        if n_episode*2 >= time_step:
            break
    # print("sum_data(split_id)", sum_data(split_id))
    split_id = cut_data_(split_id, aim_len=time_step+1)
    # print("sum_data(split_id)", sum_data(split_id))
    # print("split_id",split_id)
    (
        _obs_arr,
        _bact_id_arr,
        _lact_id_arr,
        _bact_val_arr,
        _lact_val_arr,
        _reward_arr,
        _command_arr,
        _bact_prior_arr,
    ) = ([], [], [], [], [], [], [], [])

    for split in split_id:
        n_b = split[0]
        n_e = split[1]

        _obs_arr.extend(observations[n_b : (n_e + 1)])
        _bact_id_arr.extend(
            actions_behavior_id[n_b : n_e + 1]
        ) 
        _bact_val_arr.extend(
            actions_behavior_val[n_b : n_e + 1]
        ) 
        _lact_id_arr.extend(actions_label_id[n_b : n_e + 1]) 
        _lact_val_arr.extend(
            actions_label_val[n_b : n_e + 1]
        ) 
        _reward_arr.extend(rewards[n_b : n_e + 1]) 

        _command_arr.extend(command[n_b : n_e + 1])
        if actions_behavior_prior is not None:
            _bact_prior_arr.extend(actions_behavior_prior[n_b : n_e + 1])

        

    return (
        np.array(_obs_arr),
        np.array(_bact_id_arr),
        np.array(_bact_val_arr),
        np.array(_lact_id_arr),
        np.array(_lact_val_arr),
        np.array(_reward_arr),
        np.array(_command_arr),
        np.array(_bact_prior_arr),
    )






def sum_data(split_id):
    _sum = 0
    for split in split_id:
        n_b = split[0]
        n_e = split[1]
        _sum += n_e - n_b+1
    return _sum


def _is_mazeworld_record(path):
    """Identify records by schema, with a path-name fallback for legacy data."""
    # MazeWorld records always contain the bird's-eye-view map.  This avoids
    # relying on case-sensitive mount-point names such as ``/Maze``.
    if os.path.exists(os.path.join(path, "BEVs.npy")):
        return True
    return any("maze" in part.lower() for part in os.path.normpath(path).split(os.sep))

def cut_data_(split_id,aim_len=2_000):
    while(sum_data(split_id) != aim_len):
        for split in split_id:
            n_b = split[0]
            n_e = split[1]
            if n_e - n_b >= 2:
                split_id.remove(split)
                split_id.append([n_b+1,n_e])
            if sum_data(split_id) == aim_len:
                break
    return split_id




class MazeDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        for d in directories:
            count = 0
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                        count += 1
            # file_list = os.listdir(d)
            # self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if _is_mazeworld_record(path):
            if self.folder_verbose:
                return self.__get_maze__(index), folder_name
            return self.__get_maze__(index)
        else:
            if self.folder_verbose:
                folder_name = path.split("/")[-1]
                return self.__get_procthor__(index), folder_name
            return self.__get_procthor__(index)
    
    def __len__(self):
        return len(self.file_list)
    
    def __get_maze__(self, index):
        path = self.file_list[index]

        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
            actions_label_val = np.load(path + '/actions_label_val.npy')
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0]
            assert max_t == actions_behavior_val.shape[0]
            assert max_t == actions_label_id.shape[0]
            assert max_t == actions_label_val.shape[0]
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0]

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            
            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None
    def __get_procthor__(self, index):

        path = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            # This field is optional metadata.  MazeWorld stores strings such
            # as ``rnd`` and ``exp_0.7``; only numeric priors are valid here.
            actions_behavior_prior = None
            prior_path = path + "/actions_behavior_prior.npy"
            if os.path.exists(prior_path):
                raw_prior = np.load(prior_path)
                if np.issubdtype(raw_prior.dtype, np.number):
                    actions_behavior_prior = raw_prior.astype(np.float32)

            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            elif os.path.exists(path + "/target.npy"):
                command = np.load(path + "/target.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # print(len(observations))

            percent = self.time_step / len(observations)
            if percent < 1:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = cut_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior,
                    percentage=percent,
                    time_step=self.time_step,
                )
            else:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = expend_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior, # TODO
                    percentage=percent,
                )


            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float() 
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            # print(obs_arr.shape)
            # print(self.time_step)
            obs_arr = obs_arr.permute(0, 2, 1, 3)
            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
                obs_arr[0:self.time_step+1],
                bact_id_arr[0:self.time_step], # cut the last 'end'
                lact_id_arr[0:self.time_step, 0], # lact_id_arr[0:self.time_step],
                bact_val_arr[0:self.time_step],
                lact_val_arr[0:self.time_step],
                reward_arr[0:self.time_step],
                # bact_prior_arr[0:self.time_step]
            )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None



# Test Maze Data Set
if __name__=="__main__":
    data_path = ["/home/libo/program/wordmodel/libo/for_train_word_model"]
    dataset = MazeDataSet(data_path, 1280, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, bactv, lactv, rewards, bevs = dataset[0]
    print(obs.shape, bact.shape, lact.shape, bactv.shape, lactv.shape, rewards.shape, bevs.shape)
