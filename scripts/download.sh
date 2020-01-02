#!/usr/bin/env bash

mkdir downloaded
cd downloaded
wget https://dataset-uni.s3-ap-southeast-2.amazonaws.com/fashion_testing.mp4
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
cd ..
mkdir saved_model
cd saved_model
wget https://dataset-uni.s3-ap-southeast-2.amazonaws.com/saved_model.zip
unzip saved_model.zip
rm -rf saved_model.zip