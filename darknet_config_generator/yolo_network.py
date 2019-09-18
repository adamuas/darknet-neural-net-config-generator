"""
YOLO Network Components

Includes descriptors of convolution blocks 

@author: Abdullahi S. Adamu
"""

from darknet_config_generator.yolo_connections import *
from darknet_config_generator.yolo_layers import *
from darknet_config_generator.yolo_optimizers import *
from darknet_config_generator.yolo_preprocess import *
from darknet_config_generator.common import *


""" 
YOLO NETWORK STRUCTURE

Line Comments included (see https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg)
"""
BBOX_COORDS_WCLASS_COUNT = 5

def get_pre_yolo3d_filters_count(num_classes=80, num_anchors=9, num_yolo_layers=3):
    """ returns the number of filters required for the convolution before the YOLO Object detection layer """
    return int(num_anchors/ num_yolo_layers * (BBOX_COORDS_WCLASS_COUNT + num_classes))

def _get_first_conv2d_block(start_filters=32, start_stride=1):
    """ returns first few convolution blocks """
    return [ConvolutionLayer(batch_normalize=True, filters=start_filters, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
            ConvolutionLayer(batch_normalize=True, filters=start_filters*2, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value)]

def _get_mid_conv2d_block(start_filters=32, repeats=1, activation=Activations.LEAKY_RELU.value, no_residual=False):
    """ middle convolution blocks """
    if no_residual:
        return [ConvolutionLayer(batch_normalize=True, filters=start_filters, size=1, stride=1, pad=1, activation=activation),
                ConvolutionLayer(batch_normalize=True, filters=start_filters * 2, size=3, stride=1, pad=1, activation=activation)] * repeats
    else:
        return [ConvolutionLayer(batch_normalize=True, filters=start_filters, size=1, stride=1, pad=1, activation=activation),
                ConvolutionLayer(batch_normalize=True, filters=start_filters * 2, size=3, stride=1, pad=1, activation=activation),
                SkipConnection(from_layer=-3, activation=Activations.LINEAR.value)] * repeats

def _get_downsample_conv2d(filters=128, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value):
    """downsample"""
    return [ConvolutionLayer(batch_normalize=True, filters=filters, size=size, stride=stride, pad=pad, activation=activation)]



def get_yolov3(num_classes=80, anchors=YOLO_ANCHORS, num_anchors=9):
    """ 
    returns YOLO v3 Network Architecture 

    params:
    - num_classes (int) - number of classes
    - anchors (list(int)) - List of anchors  [x_1,y_1,..x_2,y_2...x_n,y_n]
    - num_anchors (int) - number of anchors
    """
    layers = []

    layers = _get_first_conv2d_block(start_filters=32)
    layers += _get_mid_conv2d_block(start_filters=32, repeats=1)
    layers += _get_downsample_conv2d(filters=128, size=3, stride=2)

    layers += _get_mid_conv2d_block(start_filters=64, repeats=2)
    layers += _get_downsample_conv2d(filters=256, size=3, stride=2)

    layers += _get_mid_conv2d_block(start_filters=128, repeats=8)
    layers += _get_downsample_conv2d(filters=512, size=3, stride=2)

    layers += _get_mid_conv2d_block(start_filters=256, repeats=8)
    layers += _get_downsample_conv2d(filters=1024, size=3, stride=2)

    layers += _get_mid_conv2d_block(start_filters=512, repeats=4)
    

    # YOLO Layer - First Resolution
    layers += _get_mid_conv2d_block(start_filters=512, repeats=3, no_residual=True)
    layers += [ConvolutionLayer(filters=get_pre_yolo3d_filters_count(num_classes=num_classes, num_anchors=num_anchors),
                                size=1, stride=1, pad=1, activation=Activations.LINEAR.value, batch_normalize=False)]
    layers += [YOLOLayer(anchors=anchors, num_classes=num_classes, masks=[6,7,8])]

    layers += [RouteConnection(layers=[-4]),
               ConvolutionLayer(filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
               UpsampleLayer(stride=2),
               RouteConnection(layers=[-1, 61])]
    
    # YOLO Layer - Second Resolution
    layers += _get_mid_conv2d_block(start_filters=256, repeats=3, no_residual=True)
    layers += [ConvolutionLayer(filters=get_pre_yolo3d_filters_count(num_classes=num_classes, num_anchors=num_anchors),
                                size=1, stride=1, pad=1, activation=Activations.LINEAR.value, batch_normalize=False)]
    layers += [YOLOLayer(anchors=anchors, num_classes=num_classes, masks=[3,4,5])]

    layers += [RouteConnection(layers=[-4]),
               ConvolutionLayer(filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
               UpsampleLayer(stride=2),
               RouteConnection(layers=[-1, 36])]

    # YOLO Layer - Third Resolution
    layers += _get_mid_conv2d_block(start_filters=128, repeats=3, no_residual=True)
    layers += [ConvolutionLayer(filters=get_pre_yolo3d_filters_count(num_classes=num_classes, num_anchors=num_anchors),
                                size=1, stride=1, pad=1, activation=Activations.LINEAR.value, batch_normalize=False)]
    layers += [YOLOLayer(anchors=anchors, num_classes=num_classes, masks=[0,1,2])]
    
    return layers


def get_alexnet(num_classes=80):
    """
    returns AlexNet layers

    params:
    - num_classes (int) - number of classes

    returns:
    - list of layers
    """

