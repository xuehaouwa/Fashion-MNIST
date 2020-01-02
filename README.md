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

| Model | parameters | Augmentation (Random Horizontal Flips) | Accuracy | Training Time | Test Time |
| :---: | ---------- | :------------------------------------: | -------- | ------------- | --------- |
|  v1   | 0.420298M  |                                        | 90.88%   | 233.028       | 0.556     |
|  v1   | 0.420298M  |           :heavy_check_mark:           | 91.16%   | 250.486       | 0.581     |
|  v2   | 0.844362M  |                                        | 92.19%   | 263.481       | 0.600     |
|  v2   | 0.844362M  |           :heavy_check_mark:           | 92.71%   | 276.720       | 0.594     |
|  v3   |            |                                        |          |               |           |
|  v3   |            |           :heavy_check_mark:           |          |               |           |



### 



### Run On Camera Feeds or Video Files


As we are focusing on the classification task in this repo, in order to make it work with videos, we need to get help from object detection model to localize items first. We then can run the classification model on the detected region.

