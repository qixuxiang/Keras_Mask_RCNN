# Keras Mask-RCNN


## Update

This code only support to keras 2.1.5, the newest version will cause some errors. If you can fix it, feel free to send me a PR.

## Requirements
Basically, this code supports both python2.7 and python3.5, the following package should installed:
* tensorflow
* keras
* scipy
* cv2

## Out of box model to predict

I have trained a model to predict kitti. I will update a dropbox link here later. Let's see the result of predict:

<img src="http://opbocoyb4.bkt.clouddn.com/000010.png" align="center">

## Train New Dataset

to train a new dataset is also very simple and straight forward. Simply convert your detection label file whatever format into this format:

```
/path/training/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian
/path/training/image_2/000001.png,599.41,156.40,629.75,189.25,Truck
```
Which is `/path/to/img.png,x1,y1,x2,y2,class_name`, with this simple file, we don't need class map file, our training program will statistic this automatically.

## For Predict

If you want see how good your trained model is, simply run:
```
python test_frcnn_kitti.py
```
you can also using `-p` to specific single image to predict, or send a path contains many images, our program will automatically recognise that.

**That's all, help you enjoy!**
