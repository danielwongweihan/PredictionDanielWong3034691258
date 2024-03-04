from collections import OrderedDict
from random import sample
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from me292b import dynamics

import me292b.utils.tensor_utils as TensorUtils
import me292b.utils.metrics as Metrics
from me292b.utils.batch_utils import batch_utils
import me292b.utils.loss_utils as LossUtils
from me292b.utils.geometry_utils import transform_points_tensor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from me292b.models.base_models import MLPTrajectoryDecoder, RasterizedPredictionModel

import re
import pandas as pd
import os
import pickle

class BehaviorCloning(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(BehaviorCloning, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        # Exploration:
        # You can replace this unimodal decoder into multimodal decoder
        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        # Exploration:
        # You can use vectorized representation and use attention-based network structure.
        self.nets["predictor"] = RasterizedPredictionModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )
        
        self.sub_df = []
        self.sub_csv_name = []
        self.target_map = None

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["predictor"](obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )
        # Miss Rate: You can implement codes for evaluating miss rate.
        

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["predictor"](batch)
        losses = self.nets["predictor"].compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["predictor"](batch)
        losses = TensorUtils.detach(self.nets["predictor"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)
            
    def extract_id_complex(self, s):
        # Use regular expression to find the last occurrence of digits in the string
        matches = re.findall(r'\d+', s)
        return matches[-1] if matches else None
    
    def extract_senario_name(self, s):
        # Use regular expression to find the pattern before "_test_"
        match = re.search(r'^(.*?)_obs_', s)
        return match.group(1) if match else None
            
    def extract_repeated_indices(self, input_list):
        repeated_indices = {}
        for i in range(len(input_list)):
            temp = input_list[i]
            if temp not in repeated_indices:
                repeated_indices[temp] = [i]
            else:
                repeated_indices[temp].append(i)

        return repeated_indices
    
    
    def test_step(self, batch, batch_idx):
        # load the track ids
        track_id = []
        for agent_name in batch['agent_name']:
            temp = self.extract_id_complex(agent_name)
            if temp == None:
                raise 'error in agent id'
            track_id.append(int(temp))
            
        # predict the trajectory
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["predictor"](batch)
        
        # convert local coordinates to global
        world_from_agent = batch["world_from_agent"]
        predict_pos = transform_points_tensor(pout["predictions"]["positions"], world_from_agent)
        # print(pout["predictions"]["positions"][0,20])

        # load the case ids
        case_id = []
        for scene_id in batch['scene_ids']:
            temp = self.extract_id_complex(scene_id)
            if temp == None:
                raise 'error in scene_ids'
            case_id.append(float(temp)+1)
            
        # process the csv names
        csv_name = []
        for scene_id in batch['scene_ids']:
            temp = self.extract_senario_name(scene_id)
            if temp == None:
                raise 'error in scene_ids'
            csv_name.append(temp+'_sub.csv')
        
        
        # define the column names 
        column_names = ['case_id','track_id','frame_id','timestamp_ms','agent_type','track_to_predict','x1','y1']
        
        # convert the data into numpy array format.
        batch_size, pre_h, _ = predict_pos.size()
        predict_pos_np = predict_pos.view(-1,2).detach().cpu().numpy()
        case_id = np.repeat(np.asarray(case_id).reshape(batch_size,1), pre_h, axis=1)
        case_id = case_id.reshape(batch_size*pre_h,1)
        
        track_id = np.repeat(np.asarray(track_id).reshape(batch_size,1), pre_h, axis=1)
        track_id = track_id.reshape(batch_size*pre_h,1)
        
        
        # define frame id and timestamp_ms
        frame_id = np.arange(11,41)
        timestamp_ms = frame_id * 100
        
        
        # identify other attributes
        agent_type = ['car']*pre_h
        track_to_predict = [1]*pre_h
        
        if self.target_map == None:
            with open('scripts/target_map.pkl', 'rb') as f:
                self.target_map = pickle.load(f)
        
        for i in range(batch_size):
            scene_name = csv_name[i][:-8]
            target_id = np.array([case_id[i*pre_h,0], track_id[i*pre_h,0]])
            if np.any(np.all(self.target_map[scene_name] == target_id, axis=1)):
                df = pd.DataFrame(columns=column_names)
                df['case_id'] = case_id[i*pre_h:(i+1)*pre_h,0]
                df['track_id'] = track_id[i*pre_h:(i+1)*pre_h,0]
                df['x1'] = predict_pos_np[i*pre_h:(i+1)*pre_h,0]
                df['y1'] = predict_pos_np[i*pre_h:(i+1)*pre_h,1]
                df['frame_id'] = frame_id
                df['timestamp_ms'] = timestamp_ms
                df['agent_type'] = agent_type
                df['track_to_predict'] = track_to_predict
                
                self.sub_df.append(df)
                self.sub_csv_name.append(csv_name[i])
        
        return None
    
    def test_epoch_end(self, outputs) -> None:
        folder = 'submission'
        os.makedirs(folder,exist_ok = True) 
        repeated_res = self.extract_repeated_indices(self.sub_csv_name)
        for i in range(len(repeated_res)):
            print('Process the submission: ', i,'/',len(repeated_res))
            csv_name = list(repeated_res.keys())[i]
            df = pd.concat([self.sub_df[j] for j in repeated_res[csv_name]], axis=0)
            file_path = os.path.join(folder, csv_name)
            df.to_csv(file_path, index=False)



    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["predictor"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )
