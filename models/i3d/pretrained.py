#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
from . import i3d

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt'
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'


def get_num_classes(eval_type='joint'):
    num_classes = 400
    if eval_type == 'rgb600':
        num_classes = 600
    return num_classes


def get_class_labels(eval_type='joint'):
    # Getting classes from txt file
    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
    return kinetics_classes


def init_model(eval_type='joint', image_size=224, num_of_frames=16, num_classes=400):
    tf.logging.set_verbosity(tf.logging.INFO)
    graph = tf.Graph()
    with graph.as_default():
        # Initialize the inception 3Dv1 model for rgb input
        rgb_saver = None
        rgb_input = None
        flow_saver = None
        flow_input = None
        if eval_type in ['rgb', 'rgb600', 'joint']:
            # RGB input has 3 channels.
            rgb_input = tf.placeholder(
                tf.float32,
                shape=(1, num_of_frames, image_size, image_size, 3))
            with tf.variable_scope('RGB'):
                rgb_model = i3d.InceptionI3d(
                    num_classes, spatial_squeeze=True, final_endpoint='Logits')
                rgb_logits, _ = rgb_model(
                    rgb_input, is_training=False, dropout_keep_prob=1.0)
            rgb_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB':
                    if eval_type == 'rgb600':
                        rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                    else:
                        rgb_variable_map[variable.name.replace(':0', '')] = variable
            rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        # initialize the model for TLv1 flow input
        if eval_type in ['flow', 'joint']:
            # Flow input has only 2 channels.
            flow_input = tf.placeholder(
                tf.float32,
                shape=(1, num_of_frames, image_size, image_size, 2))
            with tf.variable_scope('Flow'):
                flow_model = i3d.InceptionI3d(
                    num_classes, spatial_squeeze=True, final_endpoint='Logits')
                flow_logits, _ = flow_model(
                    flow_input, is_training=False, dropout_keep_prob=1.0)
            flow_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'Flow':
                    flow_variable_map[variable.name.replace(':0', '')] = variable
            flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

        # Combine the logits of both models if joint is specified
        if eval_type == 'rgb' or eval_type == 'rgb600':
            model_logits = rgb_logits
        elif eval_type == 'flow':
            model_logits = flow_logits
        else:
            model_logits = rgb_logits + flow_logits
        model_predictions = tf.nn.softmax(model_logits)
    return {
        'graph': graph,
        'rgb_model': (rgb_input, rgb_saver),
        'flow_model': (flow_input, flow_saver),
        'output': (model_logits, model_predictions)
    }


def run_model(data, eval_type, imagenet_pretrained, graph, rgb_model, flow_model, output):
    # load samples and restore checkpoints for the models
    (rgb_input, rgb_saver) = rgb_model
    (flow_input, flow_saver) = flow_model
    (model_logits, model_predictions) = output
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')
            rgb_sample = data['rgb']
            tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            feed_dict[rgb_input] = rgb_sample
        
        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')
            flow_sample = data['flow']
            tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
            feed_dict[flow_input] = flow_sample
        
        # run the model on the loaded samples
        out_logits, out_predictions = sess.run([model_logits, model_predictions],
                                               feed_dict=feed_dict)

        out_logits = out_logits[0]
        out_predictions = out_predictions[0]

        classes = get_class_labels(eval_type)
        sorted_indeces = np.argsort(out_predictions)[::-1]
        top20_indeces = np.uint(sorted_indeces[:20])
        prediction = []
        for index in top20_indeces:
            prediction.append([out_logits[index], out_predictions[index], classes[index]])
        return prediction
