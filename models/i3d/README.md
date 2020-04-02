# I3D models trained on Kinetics

## Overview

This is the implementation of the inception-v1 I3D model described in the paper
"[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman. 

For more detail on the model architecture refer to the [deepmind implementation][deepmind_repo].

## Model Structure

### Model inputs

The model can have two types of inputs as specified by the `eval_type` parameter:
* RGB input: 3 channel rgb frames 
* Optical Flow input: 2 channel optical flow frames

The input video sampled at 25 frames per second and preprocessed according the 
preprocessing mentioned in the [deepmind repository][deepmind_repo] to produce these inputs.
This is implemented in `preprocessing.py`.

### Model Parameters

* `eval_type`: The type of model used and it determines the preprocessing used. Types allowed are
    1. `rgb`: Normal RGB model trained for kinetics 400
    2. `rgb600`: RGB model trained for kinetics 600
    3. `flow`: Optical flow model trained on kinetics
    4. `joint`: joint model of rgb and flow
* `num_of_frames`: number of frames per input sample to the model
* `image_size`: image size of input to the model
* `input_shape`: shape of the input video to the overall model (your data)
* `imagenet_pretrained`: along with the eval_type determines the checkpoint used for the weights

All these paramters are required from the client to run the model.

### Model Output

The model returns the output layer logits, the probablities for each label and the labels themselves as show in this
sample.
 
![Soccer Juggling Video](../../Test_videos/juggling_soccerball.gif)

The model gives the following output 



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


Typical flow of running the model is:
* Preprocess your video data
* Initialize the model
* Run the model using the preprocesed data as input

## Further Information

For more information about the model please refer to [its implementation][deepmind_repo]


[deepmind_repo]: https://github.com/deepmind/kinetics-i3d
