from enum import Enum, auto
import os
from absl import logging

# Defaults
YOLO_ANCHORS = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

# Constants 
NL = os.linesep

# Enumerations
class Activations(Enum):
    LINEAR = 'linear'
    RELU = 'relu'
    LEAKY_RELU = 'leaky'

class LearningRateDecayPolicy:
    SCHEDULED ='steps'

# Functions
def list_to_str(lst, space=False):
    """ convers list as a comma seperated string"""
    if space:
        base_str = ', '
    else:
        base_str = ','
        
    return base_str.join((str(c) for c in lst)).rstrip()

def anchors_to_str(anchors, space=False):
    """ converts anchors as a comma seperated string"""
    # Get Anchor Tuple
    anchor_tuple = [tuple(anchors[i-2:i]) for i in range(2,len(anchors),2)]
    # Odd case
    if (len(anchors)/2)%2 == 1:
        anchor_tuple = anchor_tuple + [tuple(anchors[-2:])]
    # base string spacing
    if space:
        base_str = ', '
    else:
        base_str = ','
    
    return ', '.join([f'{pt[0]},{pt[1]}'  for pt in anchor_tuple]).rstrip()