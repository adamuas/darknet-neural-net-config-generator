"""
Layer Descriptors

@author: Abdullahi S. Adamu
"""

from darknet_yolov3.config_generator.common import *

""" Layers """
class Layer:
    """ Layer"""
    def __init__(self):
        pass
    def export(self):
        pass

class ConvolutionLayer(Layer):
    """ Convolution Layer"""
    def __init__(self, size:int=1, filters:int=255, stride:int=3, pad:int=1,
                         activation=Activations.LEAKY_RELU.value, batch_normalize:bool=True):
        self.__HEADER__='[convolutional]'
        self.size = size
        self.filters = filters
        self.stride = stride
        self.pad = pad
        self.activation = activation
        self.batch_normalize = batch_normalize
        
    def export(self, file_obj):
        """ writes to layer to file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        if self.batch_normalize:
            file_obj.write(f'batch_normalize={int(self.batch_normalize)}{NL}')
        file_obj.write(f'size={self.size}{NL}')
        file_obj.write(f'stride={self.stride}{NL}')
        file_obj.write(f'pad={self.pad}{NL}')
        file_obj.write(f'filters={self.filters}{NL}')
        file_obj.write(f'activation={self.activation}{NL}')
        file_obj.write(NL)

        
class YOLOLayer(Layer):
    """ 
    Object Detection Layer
    
    This is normally placed at different levels of the network to perform detections at different resolutions
    """
    def __init__(self, anchors:list=YOLO_ANCHORS, num_classes:int=80, jitter:float=0.5, masks:list=[6,7,8],
                        ignore_thresh:float=0.5, truth_thresh:float=1.0, random:bool=True):
        self.__HEADER__ = '[yolo]'
        self.masks = masks
        self.anchors = anchors
        self.num_anchors = len(anchors)//2
        self.classes = num_classes
        self.jitter = jitter
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.random = int(random)
        
    def export(self, file_obj):
        """exports the layer to the given file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'mask={list_to_str(self.masks, space=False)}{NL}')
        file_obj.write(f'anchors={anchors_to_str(self.anchors)}{NL}')
        file_obj.write(f'classes={self.classes}{NL}')
        file_obj.write(f'num={self.num_anchors}{NL}')
        file_obj.write(f'jitter={self.jitter}{NL}')
        file_obj.write(f'ignore_thresh={self.ignore_thresh}{NL}')
        file_obj.write(f'truth_thresh={self.truth_thresh}{NL}')
        file_obj.write(f'random={self.random}{NL}')
        

class UpsampleLayer(Layer):
    """ Upsampling Layer"""
    def __init__(self, stride=2):
        self.__HEADER__ = '[upsample]'
        self.stride = stride
        
    def export(self, file_obj):
        """exports the layer to the given file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'stride={self.stride}{NL}')
        file_obj.write(NL)