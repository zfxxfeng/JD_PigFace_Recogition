# JDD competition of Pig Face Recogition

### Introduction
This is a repository use ResNet to achieve Pig face Recogtion. [competition](http://jddjr.jd.com/item/4).

Developed for Tensorflow 1.3
Author: [Feng zhang](https://github.com/zfxxfeng)
<br>Email: 364051598@qq.com

### Contents
1. [Data](#data)
2. [Detect](#detect)
3. [Recogition](#recogition)
3. [Train](#train)
4. [Test](#test)


### data
Download the data in the JD's [website](https://jddjr.jd.com/item/4). The origin data is 3000 Pigface images and 30 videos of 30 pigs.


### detect
way 1 Use adaboost classifier to detect all pig face in 30 videos. 3000 images are used to be the training data and train a classifier.I use this classifier to detect all pig face in 30 videos and save them in 30 folders.I don't disscuss it in detail. 
way 2 Use tensorflow object detection Api to detect the pig faces and save in 30 folders.You should try [object detection Api](https://github.com/tensorflow/models/tree/master/research/object_detection) firstly. And then, use \object_detection\cut_pig.py to get the target image.

### recogition
Model:Use 50 Resnet.you can change the layers of model to 101 or 152 or 200.Do as follows:
```
%open the train.py

%original
net_test,end_point_test = resnet_inference.resnet_v2_50(inputs=image_batch,num_classes=classes_num,keep_prob=keep_prob1,reuse=True)
%changed
net_test,end_point_test = resnet_inference.resnet_v2_101(inputs=image_batch,num_classes=classes_num,keep_prob=keep_prob1,reuse=True)
%you only need to change the number of layers
```

### Train
When you prepare 30 folders which are 30 pigs' images.Use shuffle_images_save_to_few_tfrecords.py to generate tfrecords.Do as follows:
```
%change the data dir which include 30 folders
%orginal
images,labels = data.load_data(dirname="./object_detection_api", one_hot=False, resize_pics=(300, 300))
%yours
images,labels = data.load_data(dirname="./data_dir", one_hot=False, resize_pics=(300, 300))
```
After that,Train the models.Use yours data
```
%orginal
files = tf.train.match_filenames_once("./your_data_dir/data.tfrecords-*")
```

### Test
The test data should be images.You put then in the same folders.For example:
```
%change
X_test0,Y_test0 = data.load_data(dirname="./You_test_data_dir", one_hot=True, resize_pics=(224, 224))
%run test
python test.py
```

# Have fun!! :)



