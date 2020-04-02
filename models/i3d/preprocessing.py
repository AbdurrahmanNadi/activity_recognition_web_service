import time
import numpy as np
import cv2


class InputDataPreprocessor:
    def __init__(self, image_size=224, num_of_frames=16, eval_type='joint'):
        self.image_size = image_size
        self.num_of_frames = num_of_frames
        self.eval_type = eval_type

    def preprocess_frame(self, img, input_type):
        frame = np.double(img)
        (height, width, chn) = np.shape(frame)
        if height > width:
            aspect_ratio = height / width
            height = int((self.image_size + 32) * aspect_ratio)
            width = (self.image_size + 32)
        else:
            aspect_ratio = width / height
            width = int((self.image_size + 32) * aspect_ratio)
            height = (self.image_size + 32)
        frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)
        if input_type == 'rgb':
            frame = (frame - 127.5) / 127.5
        elif input_type == 'flow':
            frame[frame > 20] = 20
            frame[frame < -20] = -20
            frame = (1 / 20) * frame
        else :
            raise TypeError('Invalid input type')
        x_margin = (height - self.image_size) / 2
        y_margin = (width - self.image_size) / 2
        if x_margin % 2 != 0:
            x_margin = np.floor(x_margin)
            x_start = int(x_margin)
            x_end = int(height - x_margin - 1)
        else :
            x_start = int(x_margin)
            x_end = int(height - x_margin)
        if y_margin % 2 != 0:
            y_margin = np.floor(x_margin)
            y_start = int(y_margin)
            y_end = int(width - y_margin - 1)
        else :
            y_start = int(y_margin)
            y_end = int(width - y_margin)
        frame = frame[x_start: x_end, y_start: y_end, :]
        return frame

    def process_input(self, frames):
        shape = np.shape(frames)
        assert shape[0] == self.num_of_frames + 1
        if self.eval_type in ['rgb', 'rgb600', 'joint']:
            rgb_data = np.zeros((1, self.num_of_frames, self.image_size, self.image_size, 3))
        if self.eval_type in ['flow', 'joint']:
            flow_data = np.zeros((1, self.num_of_frames, self.image_size, self.image_size, 2))
        old_frame = None
        for i in range(self.num_of_frames + 1):
            img = frames[i, ...]
            if i < self.num_of_frames and self.eval_type in ['rgb', 'rgb600', 'joint']:
                rgb_data[:, i, ...] = self.preprocess_frame(img, 'rgb')
            print(f'[INFO:{time.time()}] Done RGB Preprocessing on Frame{i}')
            gray_frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if i > 0 and self.eval_type in ['flow', 'joint']:
                optical_flow = InputDataPreprocessor.optical_flow_preprocessing(old_frame, gray_frame)
                flow_data[:, i-1, ...] = self.preprocess_frame(optical_flow, 'flow')
                print(f'[INFO:{time.time()}] Done Flow Preprocessing on Frame{i}')
            old_frame = gray_frame
        if self.eval_type == 'joint':
            return rgb_data, flow_data
        elif self.eval_type in ['rgb', 'rgb600']:
            return rgb_data
        elif self.eval_type == 'flow':
            return flow_data

        print(f'[INFO:{time.time()}] Done Sample Preprocessing')

    @staticmethod
    def optical_flow_preprocessing(prev_frame, next_frame):
        flow = cv2.optflow_DualTVL1OpticalFlow.create()
        return flow.calc(prev_frame, next_frame, None)
