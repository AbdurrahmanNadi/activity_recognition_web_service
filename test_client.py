#!/usr/bin/env python3
import requests
import json
import base64
import numpy as np
import cv2

base_url = "http://localhost:5000/activity_recognition/i3d/v1.0/init_model"

headers = {
  'Content-Type': 'application/json'
}


def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')


if __name__ == '__main__':
    NUM_SAMPLES = 32
    vs = cv2.VideoCapture('Test_videos/juggling_soccerball.gif')
    _, first_img = vs.read()
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    init_req = json.dumps({
                                'eval_type': 'rgb',
                                'imagenet_pretrained': 'True',
                                'image_size': 224,
                                'num_of_frames': NUM_SAMPLES
                            })
    response = requests.request("POST", base_url, headers=headers, data=init_req)
    api_url = json.loads(response.text.encode('utf8'))['API']
    image_req = json.dumps({'img': encode_img(first_img)})
    response = requests.request("PUT", api_url['upload_img'], headers=headers, data=image_req)
    i = 0
    while True:
        ret, img = vs.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_req = json.dumps({'img': encode_img(img)})
        response = requests.request("PUT", url=api_url['upload_img'], headers=headers, data=image_req)
        if response.status_code == 200:
            response = requests.request("GET", api_url['run'])
            prediction = json.loads(response.content)['prediction']
            print(f'-----------SAMPLE{i}---------------')
            for line in prediction:
                print(line)
            print('-----------------------------------')
            i += 1
        elif response.status_code == 202:
            continue
        else:
            print('Errors')
