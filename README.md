# Camera-Sensor-Fusion-Net-Extension

# CRF-Net for Object Detection (Camera and Radar Fusion Network)

This repository provides a neural network for object detection based on camera and radar data. It builds up on the work of [Keras RetinaNet](https://github.com/fizyr/keras-retinanet). 
The network performs a multi-level fusion of the radar and camera data within the neural network.
The network can be tested on the [nuScenes](https://www.nuscenes.org/) dataset, which provides camera and radar data along with 3D ground truth information.

## Requirements
- Linux Ubuntu (tested on versions 16.04 and 18.04)
- CUDA 10.0
- Python 3.5
- Docker 19.03 (only if usage in Docker is desired)
- NVIDIA Container Toolkit (nvidia docker)

## Installation
Install system wide packages:
* `sudo apt install graphviz`

All required python packages can be installed with the crfnet pip package.  We recommend to install the package in its own virtual environment. To install, run the following commands in the repository folder (crfnet).

* `pip install -e .`
* `python setup.py build_ext --inplace`

# CRF-Net Usage
The network uses camera and radar inputs to detect objects. It can be used with the nuScenes dataset and extended to other radar and camera datasets. The nuScenes dataset can be downloaded [here](https://www.nuscenes.org/download).
Pretrained weights are without distance detection provided [here](https://syncandshare.lrz.de/dl/fi9RrjqLXyLZFuhwjk9KiKjc/crf_net.h5 ) (270MB).
Pretrained weights wrt to distance detection provided : https://drive.google.com/drive/folders/1Ds_b5FqDqp6rSuK9PBu0Xxzo6svHL2gk?usp=sharing
## Start Training
1. Create your desired configuration for the CRF-Net. Start by making a copy of the default_config. 
2. We have trained with distance detection true .
3. Execute `python train_crfnet.py`. This will train a model on a given dataset specified in the configs. The result will be stored in saved_models and the logs in crfnet/tb_logs.
    * `--config <path to your config>` to use your config. Per default the config file found at ./configs/local.cfg is used.

Example usage: 
```bash
python train_crfnet.py --config configs/crf_net.cfg
```

## Evaluate Model
1. Execute `python evaluate_crfnet.py` to calculate the precision and recall values for a model on the data specified 
in the config file. The values and curves are saved onto the hard drive.
    * `--config <path to your config>` show the path of training config.
    * `--model <path to model>` model file saved from prior training
    * `--st <score trehshold>` select a custom threshold at which predictions are considered as positive.
    * `--render` to show images with predicted bounding boxes during execution
    * `--eval_from_detection_pickle` to load saved detection files from the hard drive instead of running the model to 
    evaluate it.
2. We have extracted the velocities are obtained by directly averaging the original radar data inside each bounding box.
    
Example usage: 
```bash
python evaluate_crfnet.py --model saved_models/crf_net.h5 --config configs/crf_net.cfg --st 0.5
```

## Test Model
1. Execute `python test_crfnet.py` to run a inference model on the data specified in the config file.
    * `--config <path to your config>` show the path of training config.
    * `--model <path to model>` model file saved from prior training
    * `--st <score trehshold>` select a custom threshold at which predictions are considered as positive.
    * `--render` to show images with predicted bounding boxes during execution
    * `--no_radar_visualization` suppresses the radar data in the visualization
    * `--inference` to run the network on all samples (not only the labeled ones, only TUM camra dataset)

Example usage: 
```bash
python test_crfnet.py --model saved_models/crf_net.h5 --config configs/crf_net.cfg --st 0.5
```

## Results

Here the the first value is distance, second is velocity in the front direction, third is radial velocity
![image](https://user-images.githubusercontent.com/54212099/115305308-1b1a1e80-a134-11eb-8e68-67c78119749c.png)

![image](https://user-images.githubusercontent.com/54212099/115305145-e73ef900-a133-11eb-8c34-f9debdc702a6.png)



## Files
| File | Description |
|----|----|
train_crfnet.py   | Used to train the CRF-Net. 
evaluate_crfnet.py | Used to evaluate a trained CRF-Net model on the validation set.
test_crfnet.py | Used to test a trained CRF-Net model on the test set. This script can be used to record videos.
requirements.txt | Contains the requirements for the scripts in this repository
setup.py | Installs the requirements for this repository and registers this repository in the python modules

# Contributions
[1] M. Geisslinger, "Autonomous Driving: "Object Detection using Neural Networks for Radar and Camera Sensor Fusion," Master's Thesis, Technical University of Munich, 2019

[2] M. Weber, "Autonomous Driving: Radar Sensor Noise Filtering and Multimodal Sensor Fusion for Object Detection with Artificial Neural Networks," Master’s Thesis, Technical University of Munich, 2019.

[3] Reference repo for code : https://github.com/TUMFTM/CameraRadarFusionNet/tree/master/crfnet

## Citations :

    @INPROCEEDINGS{nobis19crfnet,
        author={Nobis, Felix and Geisslinger, Maximilian and Weber, Markus and Betz, Johannes and Lienkamp, Markus},
        title={A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection},
        booktitle={2019 Sensor Data Fusion: Trends, Solutions, Applications (SDF)},
        year={2019},
        doi={10.1109/SDF.2019.8916629},
        ISSN={2333-7427}
    }
