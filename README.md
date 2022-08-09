# LIME AND RISE Explanations for Object Detection

This repository contains the implementation of the project 'Explanation for Computer Vision' in which we apply explanation techniques [LIME](https://arxiv.org/abs/1602.04938) and [RISE](https://arxiv.org/abs/1806.07421) for object detection. You can find the paper [here](./final_report.pdf). <br>
In order to see the demo of this project, you need to run ```check_explanations_coco.ipynb``` . 
Before running the notebook we need to download the coco dataset.

## Downloading the COCO Dataset
Download the COCO dataset from [here](https://cocodataset.org/#download). <br>
In order to only test the notebook without training the object detector on COCO dataset, just download [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and [2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip). Copy these file inside the ```./coco/``` folder. 

## Downloading the pretrained models
Download the pretrained models from [here](https://drive.google.com/drive/folders/1QXIL6wPGCuUyNJ-60W0x9TNR6dM0B9FN?usp=sharing) and store them in the ```./checkpoints/``` folder.
