import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import cv2
import numpy as np

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager, GeneratorBase, Logger
from airsoul.utils import noam_scheduler, LinearScheduler
from airsoul.dataloader import MazeDataSet, PrefetchDataLoader
import logging
from queue import Queue
import threading
import matplotlib.pyplot as plt
import torch.nn as nn


from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager, GeneratorBase
from airsoul.utils import noam_scheduler, LinearScheduler
from airsoul.dataloader import MazeDataSet, PrefetchDataLoader


def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

@EpochManager
class MazeEpochVAE:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "noise",
                        "kl_weight",
                        "reconstruction_error",
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys[3:])
            self.lr = self.config.lr_vae
            self.lr_decay_interval = self.config.lr_vae_decay_interval
            self.lr_start_step = self.config.lr_vae_start_step
        else:
            self.logger_keys = ["reconstruction_error", 
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys)

    def preprocess(self):
        if(self.is_training):
            self.sigma_scheduler = LinearScheduler(self.config.sigma_scheduler, 
                                                   self.config.sigma_value)
            self.lambda_scheduler = LinearScheduler(self.config.lambda_scheduler, 
                                                    self.config.lambda_value)
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_vae, verbose=self.main),
            batch_size=self.config.batch_size_vae,
            rank=self.rank,
            world_size=self.world_size
            )
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_vae_stop')):
            if(epoch_id >= self.config.epoch_vae_stop):
                return False
        return True

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr,
                behavior_act_arr, label_act_arr, rew_arr, 
                local_batch_id=-1,
                global_batch_id=-1,
                global_epoch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, seg_obs in segment_iterator(
                            self.config.seq_len_vae, self.config.seg_len_vae,
                            self.device, obs_arr):
            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()

            if(self.is_training):
                sigma = self.sigma_scheduler()
            else:
                sigma = 0
            loss = self.model.module.vae_loss(
                    seg_obs,
                    _sigma=sigma)
            losses.append(loss)
            if(self.is_training):
                syn_loss = loss["Reconstruction-Error"] + self.lambda_scheduler() * loss["KL-Divergence"]
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    reconstruction_error = loss["Reconstruction-Error"] / loss["count"],
                    kl_divergence = loss["KL-Divergence"] / loss["count"],
                    count = loss["count"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            self.sigma_scheduler(), 
                            self.lambda_scheduler(), 
                            stat_res["reconstruction_error"]["mean"], 
                            stat_res["kl_divergence"]["mean"],
                            epoch=global_epoch_id,
                            iteration=local_batch_id)
            # update the scheduler
            self.sigma_scheduler.reset(global_batch_id)
            self.lambda_scheduler.reset(global_batch_id)
        else:
            self.stat.gather(self.device,
                    reconstruction_error=loss["Reconstruction-Error"] / loss["count"], 
                    kl_divergence=loss["KL-Divergence"] / loss["count"], 
                    count=loss["count"])
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["reconstruction_error"]["mean"], 
                        stat_res["kl_divergence"]["mean"], 
                        epoch=epoch_id)

@EpochManager
class MazeEpochCausal:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
        else:
            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_stop')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main),
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size
            )

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, rew_arr,
                global_batch_id=-1,
                local_batch_id=-1,
                global_epoch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, seg_cmd, seg_obs, seg_behavior_act, seg_label_act in segment_iterator(
                                self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                                cmd_arr, (obs_arr, 1), behavior_actid_arr, label_actid_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()
            # seg_bev = seg_bev.permute(0, 1, 4, 2, 3)
            # seg_bev = seg_bev.contiguous()

            loss = self.model.module.sequential_loss(
                                    prompts = seg_cmd,
                                    observations = seg_obs,
                                    tags = None, 
                                    behavior_actions = seg_behavior_act,
                                    rewards = None,
                                    label_actions = seg_label_act, 
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    is_training=self.is_training,
                                    reduce_dim=self.reduce_dim) 
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_latent * loss["wm-latent"]
                        + self.config.lossweight_worldmodel_raw * loss["wm-raw"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                                loss_worldmodel_raw = loss["wm-raw"] / loss["count_wm"],
                                loss_worldmodel_latent = loss["wm-latent"] / loss["count_wm"],
                                loss_policymodel = loss["pm"] / loss["count_pm"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            stat_res["loss_worldmodel_raw"]["mean"], 
                            stat_res["loss_worldmodel_latent"]["mean"],
                            stat_res["loss_policymodel"]["mean"],
                            epoch=global_epoch_id,
                            iteration=local_batch_id)
        else:
            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1)
            counts = torch.cat([loss["count_pm"] for loss in losses], dim=1)

            bsz = loss_wm_r.shape[0]
            seg_num = loss_wm_l.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_worldmodel_raw=loss_wm_r[i], 
                        validate_worldmodel_latent=loss_wm_l[i], 
                        validate_policymodel=loss_pm[i],
                        count=counts[i])
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
                        epoch=epoch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    for key_name in stat_res:
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)



class MAZEGenerator(GeneratorBase):

    def __call__(self, epoch_id):
    
        folder_count = 0

        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                states = np.load(os.path.join(folder_path, 'observations.npy'))
                actions = np.load(os.path.join(folder_path, 'actions_behavior_id.npy'))

                in_context_len = self.config.in_context_len
                pred_len = self.config.pred_len
                start = self.config.start_position
                temp = self.config.temp
                drop_out = self.config.drop_out
                len_causal = self.config.seg_len_causal
                output_folder = self.config.output
                
                end = min(start + in_context_len, len(states))

                pred_obs_list = self.model.module.generate_step_by_step(
                    observations=states[start:end+1],
                    actions=actions[start:end],
                    actions_gt=actions[end:end+pred_len],
                    temp=temp,
                    drop_out = drop_out,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    in_context_len = in_context_len,
                    len_causal = len_causal,
                    n_step=pred_len
                )

                real = [states[i] for i in range(end+1, end + 1 + pred_len)] 

                pred_obs_list_with_initial = pred_obs_list
                
                
                video_folder = os.path.join(output_folder, f'video_{folder_count}')
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                video_filename = os.path.join(video_folder, f"pred_obs_video_{folder_count}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                frame_height, frame_width = pred_obs_list_with_initial[0].shape[:2]
                video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))

                for real_frame, pred_frame in zip(real, pred_obs_list_with_initial):
                    rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    concatenated_img = np.hstack((rotated_real, rotated_pred))

                    img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                    video_writer.write(img)

                video_writer.release() 

                print(f"Saved video with {len(real)} frames to {video_filename}")

                
                updated_cache = None
                print(f"Cache cleared after generating {len(real)} frames.")

                folder_count += 1  

                if folder_count >= 16:
                    print("Processed 16 folders. Stopping.")
                    break 


class general_generator(GeneratorBase): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        self.N_maze = self.config.N_maze
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"
        
        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.N_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        self.init_logger()
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
    def init_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is None):
            # self.logger_keys = self.get('logger_keys')
            if(self.logger_keys is not None and len(self.logger_keys)!=0):
                assert type(self.logger_keys) == list, \
                    f"The logger_keys must be a list of string."
                process_name = f"Generation-{self.__class__.__name__}"
                max_iter = -1
                log_file = self.log_config.log_file
                self.logger = Logger(
                        *self.logger_keys,
                        on=self.main, 
                        max_iter=max_iter,
                        use_tensorboard=self.log_config.use_tensorboard,
                        log_file=log_file,
                        prefix=f"{self.run_name}-{process_name}",
                        field=f"{self.log_config.tensorboard_log}/{self.run_name}-{process_name}")

    def __call__(self, epoch_id):
        import cv2
        batch_size = 1 # TODO
        pred_len = self.pred_len
        loss_batch = []
        cache_generate = False
        video_generate = True
        # history_cache = None
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))


            print(f"batch_id: {batch_id} processing {folder_name}")
            output_folder_path = os.path.join(self.output_root, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, behavior_act_arr, label_act_arr, rew_arr = batch_data
            obs_arr = obs_arr.permute(0, 1, 4, 2, 3) # (B, T, H, W, C) to (B, T, C, H, W)
            states = obs_arr.contiguous()
            
            actions = behavior_actid_arr.contiguous()

            history_cache = None
            loss_records = []
            pred_records = []
            real_records = []
            for checkpoint_id in range(0, self.end_position):
                end = min(checkpoint_id, len(states))
                pred_obs_list, history_cache = self.model.module.generate_states_only(
                        prompts=cmd_arr[:, end:end+pred_len],
                        current_observation=states[:, end:end+1], 
                        action_trajectory=actions[:, end:end+pred_len],
                        history_observation=None, #states[start:end],
                        history_action=None, #actions[start:end],
                        history_update_memory=False, 
                        autoregression_update_memory=False, # TOTEST
                        cache=history_cache,
                        single_batch=True,
                        history_single_step=False,
                        future_single_step=False,
                        raw_images=True,
                        need_numpy=False)
                # print(f"pred_obs_list: {pred_obs_list}")
                real = states[:, end+1:end+1+pred_len]
                mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                        loss_type="mse",
                                        gt=real, 
                                        need_cnt=True,
                                        )
                mse_loss = mse_loss/255/255
                print(f"check_point {checkpoint_id} with mse_loss: {mse_loss/cnt}")
                loss_records.append(mse_loss.detach().numpy()/cnt)  
                import copy
                if checkpoint_id in self.record_points and cache_generate == True:
                    np.save(os.path.join(output_folder_path, f"cache_{checkpoint_id}.npy"), history_cache)
                    print(f"Saved cache to {os.path.join(output_folder_path, f'cache_{checkpoint_id}.npy')}")
                pred_records.append(pred_obs_list.cpu().detach().numpy())
                real_records.append(real.cpu().detach().numpy().copy())
            loss_records = np.array(loss_records)
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)

        loss_batch = np.array(loss_batch)
        bsz = loss_batch.shape[0]
        seg_num = loss_batch.shape[1] // self.downsample_length
        valid_seq_len = seg_num * self.downsample_length
        loss_batch = np.mean(loss_batch[:, :valid_seq_len].reshape(bsz, seg_num, -1), axis=-1)
        self.stat.gather(self.device,
                validate_worldmodel_raw=loss_batch[0], 
                count=cnt)

    def epoch_end(self, epoch_id):
        stat_res = self.stat()
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is not None):
            self.logger(stat_res["validate_worldmodel_raw"]["mean"],
                    epoch=epoch_id)
        if(self.extra_info is not None):
            if(self.extra_info.lower() == 'validate' and self.main):
                if not os.path.exists(self.config.output):
                    os.makedirs(self.config.output)
                for key_name in stat_res:
                    res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                    file_path = f'{self.config.output}/result_{key_name}.txt'
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    with open(file_path, 'w') as f_model:
                        f_model.write(res_text)




