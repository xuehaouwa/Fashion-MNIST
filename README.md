# Fashion-MNIST



### Requirements

pytorch 0.4.1

torchvision 0.2.1



### Model Description

| Model | parameters | Augmentation (Random Horizontal Flips) | Accuracy | Training Time | Test Time |
| :---: | ---------- | :------------------------------------: | -------- | ------------- | --------- |
|  v1   |            |                                        |          |               |           |
|  v1   |            |           :heavy_check_mark:           |          |               |           |
|  v2   |            |                                        |          |               |           |
|  v2   | 0.844362M  |           :heavy_check_mark:           | 92.48%   | 279.82        | 0.598     |
|  v3   |            |                                        |          |               |           |
|  v3   |            |           :heavy_check_mark:           |          |               |           |



### 



### Run On Camera Feeds or Video Files


As we are focusing on the classification task in this repo, in order to make it work with videos, we need to get help from object detection model to localize items first. We then can run the classification model on the detected region.

