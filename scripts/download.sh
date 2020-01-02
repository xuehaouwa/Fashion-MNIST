#!/usr/bin/env bash

mkdir downloaded
cd downloaded
wget https://dataset-uni.s3-ap-southeast-2.amazonaws.com/fashion_testing.mp4
cd ..
wget https://dataset-uni.s3-ap-southeast-2.amazonaws.com/saved_model.zip
unzip saved_model.zip
rm -rf saved_model.zip