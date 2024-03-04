import matplotlib.pyplot as plt
import pytorch_lightning as pl
import os

from trajdata.visualization.vis import plot_agent_batch
from me292b.utils.batch_utils import batch_utils
import torch 
class VisualizeCallback(pl.callbacks.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        # Ensure the output directory exists
        output_dir = "visualize"
        os.makedirs(output_dir, exist_ok=True)
        
        # Number of batches to visualize
        num_batches_to_visualize = 10
        
        # Loaders for non-dict and dict batches
        plot_batch_loader = iter(trainer.datamodule.train_dataloader(return_dict=False))
        batch_loader = iter(trainer.datamodule.train_dataloader(return_dict=True))
        
        #move model to target devices
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_module.nets["predictor"].to(device)
        for outer_idx in range(num_batches_to_visualize):
            try:
                plot_batch = next(plot_batch_loader)
                batch = next(batch_loader)
                  # Prepare batch for model prediction
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                batch = batch_utils().parse_batch(batch)
                
            except StopIteration:
                break  # Stop if no more batches
            
            # Assuming you might want to visualize more than one sample per batch
            num_samples_to_visualize = 10  # For example, visualize up to 5 samples per batch
            
            for batch_idx in range(num_samples_to_visualize):
                # Model prediction
                
                pred = pl_module.nets["predictor"](batch)
                pred_samples = pred["predictions"]["positions"].detach().cpu().numpy()  # Shape: B, sample, T, 2
                
                
                fig, ax = plt.subplots()
                ax = plot_agent_batch(plot_batch, batch_idx=batch_idx, show=False, close=False, ax=ax)
                ax.plot(pred_samples[batch_idx, :, 0], pred_samples[batch_idx, :, 1], label=f"prediction")
                ax.set_title(f"Batch {outer_idx}")
                plt.savefig(os.path.join(output_dir, f"{outer_idx}_{batch_idx}.png"))
                plt.close(fig)