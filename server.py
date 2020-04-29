#!/usr/bin/env python3
import os
import sys
import time
import base64
import threading
import random
import cv2 as cv
import numpy as np
from flask import Flask, request, redirect, jsonify, url_for, abort, make_response
from PretrainedModel import I3DPretrainedModel


random.seed(time.time())
app = Flask(__name__)
app.config['MAX_NUM_USERS'] = 100
app.config['TIME_OUT'] = 6000

active_users = {}


def decode_img(img_str):
    img_bytes = bytes(img_str, 'utf-8')
    img_buff = base64.b64decode(img_bytes)
    img_jpg = np.frombuffer(img_buff, dtype=np.uint8)
    img = cv.imdecode(img_jpg, cv.IMREAD_COLOR)
    return img


def update_user_timeout():
    print('some')
    for _id in active_users.keys():
        user = active_users[_id]
        user['time'] += 1
        if user['time'] > app.config['TIME_OUT']:
            redirect(url_for('clean_session', user_id=_id, _external=True))


class Job(threading.Thread):
    def __init__(self, interval, execute, *args, **kwargs):
        threading.Thread.__init__(self)
        self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        self.execute = execute
        self.args = args
        self.time_up
        self.kwargs = kwargs

    def stop(self):
        self.stopped.set()
        self.join()

    def run(self):
        while not self.stopped.wait(self.interval.total_seconds()):
            self.execute(*self.args, **self.kwargs)


given_ids = []


@app.route('/activity_recognition/i3d/v1.0/init_model', methods=['POST'])
def init_model():
    if len(given_ids) < 100:
        if not request.json:
            abort(400)
        params = request.json
        try:
            params['imagenet_pretrained'] = True if params['imagenet_pretrained'] == 'True' else False
        except KeyError:
            pass
        user_id = random.randint(0, sys.maxsize)
        while user_id in given_ids:
            user_id = random.randint(0, sys.maxsize)
        given_ids.append(user_id)
        model = I3DPretrainedModel(**params) # creates a model instance with the given paramters
        print(f'[INFO: {time.time()}] Model Created With ID {user_id}')
        active_users[user_id] = {'model': model,
                                 'time': 0,
                                 'sample_indx': 0,
                                 'sample_shape': (model.num_of_frames + 1, None),
                                 'last_prediction': '',
                                 'sample': None}
        return make_response(jsonify({
            'API':
            {
                'run': url_for('run_model', user_id=user_id, _external=True),
                'upload_img': url_for('add_image', user_id=user_id, _external=True)
            }
        }), 201)
    else:
        abort(503)


@app.route('/activity_recognition/i3d/v1.0/prediction/<int:user_id>', methods=['GET'])
def run_model(user_id):
    if user_id not in given_ids:
        abort(404)
    sample = active_users[user_id]['sample']
    model = active_users[user_id]['model']
    model.preprocess_data(sample)
    cwd = os.getcwd()
    os.chdir('models/i3d')
    try:
        active_users[user_id]['prediction'] = model.forward()
    except Exception as e :
        os.chdir(cwd)
        return make_response(jsonify({'status': 'failed',
            'prediction': None, 'err_msg':str(e)}), 500)
    os.chdir(cwd)
    active_users[user_id]['sample'][...] = 0
    active_users[user_id]['sample_indx'] = 0
    prediction = [f'{line[0]}, {line[1]}, {line[2]}\n' for line in active_users[user_id]['prediction']]
    active_users[user_id]['last_prediction'] = prediction
    return make_response(jsonify({'status': 'success',
                                  'prediction': prediction}), 200)


@app.route('/activity_recognition/i3d/v1.0/upload_image/<int:user_id>', methods=['PUT'])
def add_image(user_id):
    if user_id not in given_ids:
        abort(404)
    req = request.json
    if not req or 'img' not in req:
        abort(400)
    img = decode_img(req['img'])
    user = active_users[user_id]
    if not user['sample_shape'][1]:
        user['sample_shape'] = (user['sample_shape'][0], *img.shape)
        user['sample'] = np.zeros(user['sample_shape'], dtype=np.uint8)
    input_shape = user['sample_shape'][1:]
    num_of_frames = user['sample'].shape[0]
    if not (img.shape[0] == input_shape[0] and img.shape[1] == input_shape[1] and
            img.shape[2] == input_shape[2]):
        img = cv.resize(img, (input_shape[1], input_shape[0]), cv.INTER_LINEAR)
    if user['sample_indx'] > num_of_frames:
        abort(503)
    user['sample'][user['sample_indx'], ...] = img
    user['sample_indx'] += 1
    if user['sample_indx'] >= num_of_frames:
        return make_response(jsonify({'Status: ': 'finished',
                                      'sample_indx': user['sample_indx'],
                                      'last_prediction': user['last_prediction']}), 200)
    else:
        return make_response(jsonify({'Status: ': 'not_finished',
                                      'sample_indx': user['sample_indx'],
                                      'last_prediction': user['last_prediction']}), 202)


@app.route('/activity_recognition/i3d/v1.0/cleanup/<int:user_id>', methods=['DELETE'])
def clean_session(user_id):
    pass


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'Error': 'Bad Request'}), 400)


@app.errorhandler(404)
def bad_request(error):
    return make_response(jsonify({'Error': 'Resource not found'}), 404)


@app.errorhandler(503)
def bad_request(error):
    return make_response(jsonify({'Error': 'Service is not available, please try later'}), 503)


if __name__ == "__main__":
    app.run(debug=False)
