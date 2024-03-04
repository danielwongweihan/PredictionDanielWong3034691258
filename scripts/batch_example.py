import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_agent_batch
import matplotlib.pyplot as plt
import os

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run the trajectory data processing.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset directory.")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    dataset_path = args.dataset_path

    dataset = UnifiedDataset(
        desired_data=["val"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(0.9, 0.9),
        future_sec=(3.0, 3.0),
        only_predict=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=0,
        obs_format="x,y,xd,yd,xdd,ydd,s,c",
        verbose=True,
        data_dirs={
            "interaction_single": dataset_path
        },
        save_index=False,
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=16,
    )
    
    folder = 'batch_example_visualization'
    os.makedirs(folder,exist_ok = True) 
    i = 0
    for batch in tqdm(dataloader):
        i += 1
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax = plot_agent_batch(batch, batch_idx=0, ax=ax, close=False)
        plt.savefig(folder+f"/{i}_interaction_batch_example.png")
        if i >= 20:
            break

if __name__ == "__main__":
    main()
