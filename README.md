# NeRF

This is an implementation of NeRF.
**[Paper](https://arxiv.org/abs/2003.08934)**

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/samcao416/NeRF_vrvc.git
cd NeRF_vrvc
```

- Install [PyTorch](http://pytorch.org) and other dependencies using: 
```
conda create -n nerf python=3.8.5
conda activate nerf    
conda install pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install imageio matplotlib
pip install yacs kornia
pip install tensorboard
```


### Datasets
The synthetic datasets can be downloaded from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

### To Run the Program
- Create a folder named configs, copy ./config_example/hotdog.yml into the configs folder.
- Change the training directory(DATASETS:TRAIN) and output directory(OUTPUT_DIR) as you want. 
- Go to the 'tools' folder in terminal and run python train_net.py -c <../configs/hotdog.yml>

### To Monitor the trainning process
- After you can see the PSNR and loss, you can start a new terminal, and cd to your output directory.
- Run tensorboard --logdir . 
- Now you can watch the PSNR and loss curve as well as the training images in the webpage.
