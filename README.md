# Introduction
## Story
I have three children. The yongest child is still a baby, but I don't have time to take care of her when I'm busy with elder siblings. In fact, for newborn babies, even just turning over and lying on their stomachs can be life-threatening due to suffocation. Therefore, it is very important to keep an eye on your baby to make sure he or she doesn't stay prone all the time.

## About my project
I have created an application that monitors wheter the baby stays prone or supine on my behalf. This will make every day safe for the baby.  
![Videotogif](https://user-images.githubusercontent.com/94183002/145854176-cdd3c6c4-6128-4479-866e-83a504449a31.gif)  
[Youtube](https://youtu.be/1ZFuGasA1_s)

# How to run
## Requirements
I confirm by the following environment.
* Hardware
  * Jetson Nano Development Kit
  * Logicool Web Camera C270n
* Software
  * JetPack 4.6

## Advance preparations
Before starting this project, please set up your Jetson Nano according to the following information.
* Install JetPack SDK
* Detailed setup videos
* Run the hello docker container 

## Clone github
```bash
$ cd
$ git clone https://github.com/Sho-N/BabyWatcher.git
```

## Run
This project runs on the docker container as described in [Hello AI World](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md).  
Show how to run the sample.
``` bash
# run the container
$ cd ~/jetson-inference
$ docker/run.sh --volume ~/jetson-inference/BabyWatcher:/BabyWatcher

# run sample in the container
$ sh /BabyWatcher/inference_sample.sh 

# set a video to monitor your baby
```

# Data Collection
## Take a video
To prepare data for training, I took a video of my little baby with a camera. We took some videos with different clothes and shooting locations.

## Annotations
We used CVAT as an annotation tool. It is easy to annotate even videos and can be exported in Pascal VOC format.  
![image](https://user-images.githubusercontent.com/94183002/145717191-976ce64b-0582-4ad7-896e-dc9d35830a04.png)


## Merge datasets
For easy training, you can merge multiple datasets in Pascal VOC format into one. After that, the programm automatically split the dataset into training/validation.
Please set multiple dataset created by CVAT in "src_dir".
```
merge_datasets.py --src_dir=./data/multiple_dataset --dst_dir=./data/merged_dataset
```


# Train
I ran a transfer learning on SSD-Mobilenet by using the merged dataset. However, it took too much time, so the model on GitHub have learned on a Google Colaboratory.
```
cd jetson-inference
docker/run.sh
cd python/training/detection/ssd
python3 train_ssd.py
 --dataset-type=voc
 --data=data/BabyWatcher/merged_dataset/
 --model-dir=models/BabyWatcher/
 --batch-size=16
 --workers=1
 --epochs=100
```

After transfer learning, convert the model to onnx format.
```
python3 onnx_export.py --model-dir='models/BabyWatcher/'
```


# Future Direction
Prepare a lot of learning data
* Children grow up quickly
* Need data from other children
* Need more clothes, backgrounds, etc.

Data expansion
* Data augmentations
* and other ways...

Alerts
* I want to notify in some way after a certain period of time in a prone position.
* I want to notify before baby goes out of camera frame. This is the main purpose why I use detection, not classification.


# References
* [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
