#!/usr/bin.env python3
import abc
import numpy as np
from models.i3d.preprocessing import InputDataPreprocessor
import models.i3d.pretrained as pretrained


class PretrainedModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def preprocess_data(self, data):
        pass

    @abc.abstractmethod
    def forward(self):
        pass


class I3DPretrainedModel(PretrainedModel): #the model
    def __init__(self, eval_type='rgb', image_size=224, num_of_frames=16, imagenet_pretrained=True):
        self.eval_type = eval_type
        self.image_size = image_size
        self.num_of_frames = num_of_frames
        self.imagenet_pretrained = imagenet_pretrained
        classes = pretrained.get_num_classes(eval_type)
        self.model = pretrained.init_model(eval_type, image_size, num_of_frames, classes)
        self.preprocessor = InputDataPreprocessor(image_size, num_of_frames, eval_type)
        self.data = None
        self.prediction = False

    def preprocess_data(self, sample):
        self.data = self.preprocessor.process_input(sample)

    def forward(self):
        self.prediction = pretrained.run_model(self.data, eval_type=self.eval_type,
                                                           imagenet_pretrained=self.imagenet_pretrained,
                                                           **self.model)
        return self.prediction
