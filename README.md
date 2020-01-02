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

- v1: two 
- v2:
- v3



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

Training Process of the above listed models:

- During training, the learning rate will be dropped with a factor of 0.3 at epoch 40.
- For `v3_da_2`, the epoch and initial learning rate are set to 80 and 0.015.
- For all other models, the epoch and initial learning rate are 60 and 0.025.



### Run On Camera Feeds or Video Files


As we are focusing on the classification task in this repo, in order to make it work with videos, we need to get help from object detection model to localize items first. We then can run the classification model on the detected region.

