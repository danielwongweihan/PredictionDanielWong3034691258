import argparse
import sys
import os
# Set your CUDA visible devices:
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,'

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from me292b.utils.log_utils import PrintLogger
#import trajdata module
from me292b.data.trajdata_datamodules import UnifiedDataModule
from me292b.models.algos import BehaviorCloning

#Configs
from me292b.configs.base import BehaviorCloningConfig, TrainConfig

#callback function
from me292b.utils.visualize_utils import VisualizeCallback
def main(train_cfg, algo_cfg, debug=False, root_dir = "", mode = 'train', checkpoint=None):
    pl.seed_everything(train_cfg.seed)
      
    # -----Dataset------------------
    datamodule = UnifiedDataModule(algo_config=algo_cfg, train_config=train_cfg)

    # ------ Model ---------------------
    if checkpoint is not None:
        model = BehaviorCloning.load_from_checkpoint(
                algo_config=algo_cfg, 
                modality_shapes=datamodule.modality_shapes,
                checkpoint_path=checkpoint
            )
    else:
        model = BehaviorCloning(
            algo_config = algo_cfg,
            modality_shapes=datamodule.modality_shapes
        )
    
    # a ckpt monitor to save at fixed interval
    train_callbacks=[]
    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{root_dir}/checkpoints",
        filename="iter{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        monitor=None,
        every_n_train_steps=1000,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)
    
    
    # Logging
    logger = None

    logger = TensorBoardLogger(
        save_dir=root_dir, name=None, sub_dir="logs/"
    )
    print("Tensorboard event will be saved at {}".format(logger.log_dir))
   
    
    # create root_dir if it does not exist
    os.makedirs(root_dir, exist_ok=True)

    #if debug, visualize valdiation results
    if debug:
        train_callbacks.append(VisualizeCallback())

    # Train
    # Common trainer options
    trainer_options = {
        "default_root_dir": root_dir,
        # Checkpointing
        "enable_checkpointing": train_cfg.save.enabled,
        
        # Logging
        "logger": logger,
        "log_every_n_steps": train_cfg.logging.log_every_n_steps,
        
        # Training
        "max_steps": train_cfg.training.num_steps,
        
        # Validation
        "val_check_interval": train_cfg.validation.every_n_steps,
        "limit_val_batches": train_cfg.validation.num_steps_per_epoch,
        
        # All callbacks
        "callbacks": train_callbacks,
        # Number of gpus. Not support multi-gpu training.
        "gpus": 1,
    }

    # Add debug-specific options
    if debug:
        trainer_options.update({
            "overfit_batches": 1,
            # Uncomment if you want to use fast_dev_run for a quick complete run (training+validation+test)
            "fast_dev_run": True,
            "val_check_interval": 1
        })

    # Initialize the trainer with the configured options
    trainer = pl.Trainer(**trainer_options)

    if mode == 'train':
        trainer.fit(model=model, datamodule=datamodule)
    elif mode == 'val':
        trainer.validate(model=model, datamodule=datamodule)
    elif mode == 'test':
        trainer.test(model=model, datamodule=datamodule)
    else:
        raise ValueError(f'{mode} is not a valid network mode.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(optional) if provided, load from checkpoint",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Debug mode, suppress wandb logging, etc."
    )
    
    parser.add_argument(
        "--mode", 
        choices=['train', 'val', 'test'],
        default = 'train',
        help="Network mode, 'train','val' or 'test'."
    )
    

    args = parser.parse_args()

    # Set global variables 
    # Instantiate configuration objects
    train_cfg = TrainConfig()
    train_cfg.dataset_path = args.dataset_path
    algo_cfg = BehaviorCloningConfig()
    

    main(train_cfg, algo_cfg, debug=args.debug,root_dir = args.output_dir, mode = args.mode, checkpoint=args.checkpoint)
