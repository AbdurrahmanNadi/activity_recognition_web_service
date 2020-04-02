#Activity Recognition Web Service

##Overview

This repository contains the source for a REST API web service that performs human activity recognition
based on state of the art deep learning models. This app is still under development and will continue to be update regularly.

###Features in the current version

Currently the service can accept an image stream from the client build samples according to the client pre configured parameters and call a model to run inference on the samples.

##Setup 

The service can be setup in two ways either as a docker image for deployment or testing reasons
or from source for further development.

For both installations you will need to:
 * Clone this repo `$ git clone https://github.com/AbdurrahmanNadi/activity_recognition_web_service.git`
 * run the `repo_setup.sh` script to download the pretrained checkpoints.

###Docker image

* Install [docker](https://docs.docker.com/install/) on your machine
* run `$ docker build <the root of this repo> -t <image_name:tag>`
* Running the docker image will launch the service on the default port 5000 so you need to expose this port to your client

###Source Install

####Requirements:

1. Ubuntu Machine
2. Python 3
3. [OpenCV](https://github.com/opencv/opencv.git)
4. [Tensorflow version 1.15](https://www.tensorflow.org/install/pip)
5. [dm-sonnet version < 2](https://github.com/deepmind/sonnet)
6. [tensorflow-probability version < 0.9.0](https://github.com/tensorflow/probability)
7. [Flask](https://github.com/pallets/flask/)


###Test

On your client setup a python environment with opencv and simply run 
the test script you need at least the `test_client.py` script and the `Test_videos` directory
on your client.

`$ python3 test_client.py`

You should get these results.

  output logits     |       probability        |           label
:-----------------: | :----------------------: | :-----------------------:
19.75971221923828   |  0.6281151175498962      | kicking soccer ball
18.561365127563477  |  0.18949760496616364     | playing tennis
16.936567306518555  |  0.03732183575630188     | juggling soccer ball
16.709701538085938  |  0.029746539890766144    | throwing ball
16.463151931762695  |  0.023246699944138527    | kicking field goal
16.3748836517334    |  0.021282708272337914    | triple jump
15.899219512939453  |  0.013226610608398914    | playing volleyball
15.78715705871582   |  0.011824436485767365    | hurdling
15.72299575805664   |  0.01108959224075079     | passing American football (in game)
15.449047088623047  |  0.008432204835116863    | shooting goal (soccer)
15.232433319091797  |  0.006789956707507372    | dodgeball
14.79544734954834   |  0.004386179614812136    | riding unicycle
14.736922264099121  |  0.0041368454694747925   | playing badminton
14.032987594604492  |  0.002046229550614953    | long jump
13.781754493713379  |  0.0015916412230581045   | cartwheeling
13.742475509643555  |  0.0015303350519388914   | high kick
13.10960578918457   |  0.0008127083419822156   | passing American football (not in game)
12.78390884399414   |  0.0005867949221283197   | vault
12.425848007202148  |  0.00041018755291588604  | catching or throwing frisbee
12.40009593963623   |  0.00039975924300961196  | playing squash or racquetball



##Models Supported

###Kinetics pretrained I3D model

Based on google deepmind implementation of the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman. The paper was posted on arXiv in May 2017. 

The implemented model is based on the inception-v1 I3D model as stated in the paper
and pretrained on the kinetics data set split. For more information about the model 
you can visit [the deepmind github repo][deepmind_repo]

####Model Details

These are the implementation details related to the service for more details on the model structure you can refer to the [deepmind repo][deepmind_repo]
* It can accept rgb or optical flow samples or both and averaging their output logits
* The model is built as a static tf graph and can't change its configuration during running.
* The input videos must be the same size for the same client session
* The model provides its own preprocessing for input samples.
For more details please refer to [the model README][i3d]

######You can find all implemented models inside the models directory


##Web Service

###The Underlying REST API

METHOD         | Request URL             | Function
-------------- | :---------------------- | -----------
POST           | <base_uri>/init_model   | creates a model
PUT            | <base_uri>/upload_image | uploads an image
GET            | <base_uri>/prediction   | returns a prediction


##License

This software is licensed under the Apache License Version 2.0 and contains open source code from other repositories
that support similar kind of licensing

##Questions

For any questions please direct them to [Abdelrahman Nadi](abdurrahman.naddie@gmail.com)


[deepmind_repo]: https://github.com/deepmind/kinetics-i3d
[i3d]: https://github.com/AbdurrahmanNadi/activity_recogntion_web_service/models/i3d