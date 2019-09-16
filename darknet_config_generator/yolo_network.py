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
    """ returns yolo v3 network architecture """
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



""" Manually created network below"""
YOLO_LAYERS = [
    # Convolution group 1
    ConvolutionLayer(batch_normalize=True, filters=32, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=64, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=32, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=64, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),
    
    # Convolution group 2
    ConvolutionLayer(batch_normalize=True, filters=128, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=64, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=128, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),
    
    # Convolution group 3
    ConvolutionLayer(batch_normalize=True, filters=64, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=128, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),
    
    # Convolution Group 4
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),
    
    # Convolution Group 5
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),
    
    # Convolutions (L-163)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L-183)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 204)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 224)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 244)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 264)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 286) - Downsample
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 315)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 336)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 357)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 377)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 398)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 419)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 439)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 461) - Downsample
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=2, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 489)
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 509)
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 529)
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    SkipConnection(from_layer=-3, activation=Activations.LINEAR.value),

    # Convolutions (L 551)
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=1024, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    
    # YOLO LAYER (L 599)
    ConvolutionLayer(batch_normalize=False, filters=255, size=1, stride=1, pad=1, activation=Activations.LINEAR.value),
    YOLOLayer(masks=[6,7,8], anchors=YOLO_ANCHORS, num_classes=80, jitter=0.3, ignore_thresh=0.7, truth_thresh=1, random=1),
    RouteConnection(layers=[-4]),

    # (L 621)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    UpsampleLayer(stride=2),
    RouteConnection(layers=[-1, 61]),

    # (L 637)
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=512, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),

    # YOLO LAYER  (L 693)
    ConvolutionLayer(batch_normalize=False, filters=255, size=1, stride=1, pad=1, activation=Activations.LINEAR.value),
    YOLOLayer(masks=[3,4,5], anchors=YOLO_ANCHORS, num_classes=80, jitter=0.3, ignore_thresh=0.7, truth_thresh=1, random=1),
    RouteConnection(layers=[-4]),

    # (L 708)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    UpsampleLayer(stride=2),
    RouteConnection(layers=[-1, 36]),

    # (L 724)
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=128, size=1, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),
    ConvolutionLayer(batch_normalize=True, filters=256, size=3, stride=1, pad=1, activation=Activations.LEAKY_RELU.value),

    # YOLO LAYER  (L 772-789)
    ConvolutionLayer(batch_normalize=False, filters=255, size=1, stride=1, pad=1, activation=Activations.LINEAR.value),
    YOLOLayer(masks=[0,1,2], anchors=YOLO_ANCHORS, num_classes=80, jitter=0.3, ignore_thresh=0.7, truth_thresh=1, random=1)
]
