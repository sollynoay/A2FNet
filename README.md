# A2FNet  
This repository contains the source code for our paper:  
[Elevation Angle Estimation in 2D Acoustic Images Using Pseudo Front View](http://www.robot.t.u-tokyo.ac.jp/~yamashita/paper/A/A162Final.pdf)    
RAL and ICRA2021
### Libraries
```
pytorch 1.2.0
torchvision 0.40
tensorboardX
```  
### Train
```
python train.py --train_path to_your_train_label --test_path to_your_validate_label
```
During training, using front depth map instead of inverse front depth map can achieve better results
# Dataset
The simulation dataset used in this paper can be download from the following links.  
[water tank](https://drive.google.com/file/d/1GIkl_PlVbrqaCWxCzARVR4uCt6wzuq8H/view?usp=sharing)  
[floating object](https://drive.google.com/file/d/1zpviswi8ZgqrFrXDNCaeWle3ZYZC1ZUC/view?usp=sharing)

The structure of the dataset is as follows.
```
├── water-tank
│   ├── bricks
        ├── sfront.txt //inverse front depth map without quantization loss
        ├── resizeimg.png // synthetic acoustic image
        ├── ele_resize.png // ground truth of elevation angle
        ├── front_resize.png // inverse front depth map for visualization only
│   └── cylinder
```

# Simulator
For the simulator used to generate synthetic datasets, check [here](https://github.com/sollynoay/Sonar-simulator-blender).
# Evaluation
How to evaluate depth map is shown in eval.ipynb. 
