### Start with prediction project
 - Ubuntu system
 - Dataset: [Interaction dataset](https://challenge.interaction-dataset.com)
 - Task: Marginal trajectory prediction

### Notice
This repo builds upon foundational work by NVlabs, specifically leveraging the following repositories:

- [Traffic Behavior Simulation](https://github.com/NVlabs/traffic-behavior-simulation/tree/main)
- [Trajdata](https://github.com/NVlabs/trajdata)


### Download the dataset
You can download the dataset from the [Interaction dataset](https://challenge.interaction-dataset.com) or from [Google Drive](https://drive.google.com/file/d/1jeTI1BSukwdirmYN9nLm9691b0mx3TqJ/view?usp=share_link)

### Environment Setup

Follow these steps to create and activate a new Conda environment:

1. **Create the Conda Environment**:
    ```sh
    conda create -n trajdata_interaction python=3.9
    ```

    This command creates a new environment named `trajdata_interaction` with Python 3.9.

2. **Activate the Environment**:
    ```sh
    conda activate trajdata_interaction
    ```

    Activating the environment configures your terminal session to use the Python version and libraries installed in `trajdata_interaction`.

3. **Install `trajdata`**:
    ```sh
    pip install trajdata
    ```

    This command installs the base `trajdata` package.

4. **Install the `interaction` Module**:
    ```sh
    pip install "trajdata[interaction]"
    ```

    Including the `interaction` module adds functionalities tailored for interaction-aware traffic simulation.
5. **Install requirements for this base model**:
    ```sh
    pip install -e .
    ```
6. **Install Pytorch**:
    ```sh
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113 
    ```



### Command to train, debug and test the models
#### Activate your conda env
```sh
conda activate trajdata_interaction
```

#### Visualize Interactiondataset data
```sh
python scripts/batch_example.py --dataset_path <Your path to Interaction Dataset>
```
For example,
```sh
python scripts/batch_example.py --dataset_path /home/root/root/me292/prediction/interaction/INTERACTION-Dataset-DR-single-v1_2  
```

#### Train your model
```sh
python scripts/train.py --dataset_path <Your path to Interaction Dataset> --output_dir experiments/<Your experiment name>
```
For example,
```sh
python scripts/train.py --dataset_path /home/root/root/me292/prediction/interaction/INTERACTION-Dataset-DR-single-v1_2 --output_dir experiments/base_model
```

#### Debug your model. It will generate 100 images with your predicted trajectories.
```sh
python scripts/train.py --dataset_path <Your path to Interaction Dataset> --output_dir experiments/<Your experiment name> --checkpoint <Your path to ckpt> --debug
```
For example,
```sh
python scripts/train.py --dataset_path /home/root/root/me292/prediction/interaction/INTERACTION-Dataset-DR-single-v1_2  --output_dir experiments/base_model --checkpoint experiments/base_model/checkpoints/iter35000.ckpt --debug
```
And the images will be saved in **visualize** folder.

#### Test your model
```sh
python scripts/train.py --dataset_path <Your path to Interaction Dataset> --output_dir experiments/<Your experiment name> --checkpoint <Your path to ckpt> --mode test
```
For example,
```sh
python scripts/train.py --dataset_path /home/yixiao/yixiao/me292/prediction/interaction/INTERACTION-Dataset-DR-single-v1_2  --output_dir experiments/base_model --checkpoint experiments/base_model_1/checkpoints/iter35000.ckpt --mode test
```

### Possible approach to improve the base model
- Multimodal decoder: line ~30, me292b/models/algos.py, we recommend to look at MultiModal_trajectory_loss in me292b/utils/loss_utils.py
- Attention-based network structure: line ~50, me292b/models/algos.py
- Training configuration: line ~30, me292b/configs/base.py
- Network structure: line ~70, me292b/configs/base.py
- Whether to use dynamic layer: line ~90, me292b/configs/base.py
- Training loss: line ~330, me292b/models/base_model.py
- Dataloader parameters: line ~70, me292b/data/trajdata_datamodules.py