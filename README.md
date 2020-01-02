# Fashion-MNIST



### Requirements

pytorch 0.4.1
torchvision 0.2.1
(For Saving Video)
scikit-video
(For Object Detection)
tensorflow 1.8.0
keras 2.1.0
imageai
(For Visualization)
k-vision



### Model Description

There are three simple CNN models in `models`:

- v1: two conv layers and two FC layers , without BatchNormalization
- v2: two conv layers (larger channel number than v1) and two FC layers , with BatchNormalization
- v3: three conv layers and three FC layers , with BatchNormalization



The performance of these models are listed below:

|  Model  | parameters | Augmentation (Random Horizontal Flips) | Accuracy | Training Time | Test Time |
| :-----: | :--------: | :------------------------------------: | :------: | :-----------: | :-------: |
|   v1    | 0.420298M  |                                        |  90.88%  |    233.028    |   0.556   |
|  v1_da  | 0.420298M  |           :heavy_check_mark:           |  91.16%  |    250.486    |   0.581   |
|   v2    | 0.844362M  |                                        |  92.19%  |    263.481    |   0.600   |
|  v2_da  | 0.844362M  |           :heavy_check_mark:           |  92.71%  |    276.720    |   0.594   |
|   v3    | 30.076234M |                                        |  94.28%  |    874.983    |   1.179   |
|  v3_da  | 30.076234M |           :heavy_check_mark:           |  94.39%  |    889.963    |   1.215   |
| v3_da_2 | 30.076234M |           :heavy_check_mark:           |  94.55%  |   1189.720    |   1.223   |

Note the training and test time are benchmarked using a GTX-1080 GPU. Training time is the total time of all epochs.



Training Process of the above listed models:

- During training, the learning rate will be dropped with a factor of 0.3 at epoch 40.
- For `v3_da_2`, the epoch and initial learning rate are set to 80 and 0.015.
- For all other models, the epoch and initial learning rate are 60 and 0.025.



### How to Run

- Train Yourself
  1. `mkdir saved_model`  create path to save trained models
  2. `python train.py`  you can change different models, learning rate and others by changing input args. There are some examples in `scripts/train.sh`
- Use Pretrained Models
  1. `bash scripts/download.sh` this will downloaded all 7 trained models (listed in the above table) and a short video (for testing the following Run on Video Files).
  2. `python test.py`  same as `train.py`, you can select different predicted models.



### Run on Camera Feeds or Video Files


As we are focusing on the classification task in this repo, in order to make it work with videos, we need to get help from object detection model to localize items first. We then can run the classification model on the detected region.

