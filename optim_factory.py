import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    """
    Desgined to map different components of ViT to their respective layer indices.
    Useful in learning rate scheduling strategies like layer decay.
    """


    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0 # for these 3 tokens, although not a layer, we map them a layer 0 as we want to control the values update of these tokens
    elif var_name.startswith("patch_embed"):
        return 0 # patch embedding layer responsible for converting image patches into image embeddings
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1 # assigning a last layer number to the relative position biases
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1]) #extracts layer number from vairable name eg. 'blocks.0.attn', adds 1 
        return layer_id + 1 
    elif var_name.startswith("transformer.resblocks"):
        layer_id = int(var_name.split(".")[2])
        return layer_id + 1
    elif var_name in ("class_embedding", "positional_embedding", "temporal_positional_embedding"):
        return 0
    elif var_name.startswith("conv1"):
        return 0
    else:
        return num_max_layer - 1
    


def get_num_layer_for_vim(var_name, num_max_layer):
    """
    For the ViM case.
    """

    if var_name in ("cls_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed") or var_name.startswith("head") or var_name.startswith("norm_f"):
        return 0
    elif var_name.startswith("layers"): # for VideoMamba
        layer_id = int(var_name.split('.')[1])
        if 'mixer' in var_name:
            return layer_id + 1
        else:
            return 0
    else:
        return num_max_layer
    

class LayerDecayValueAssigner(object):
    """
    This class sets different learning rates or scaling factors for different layers.
    """

    def __init__(self, values):
        self.values = values # list of different scaling values in order of the layers

    def get_scale(self, layer_id):
        return self.values[layer_id]
    
    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))
    


def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layers=None, get_layer_scale=None
):
    """
    Divides the model parameters into groups based on their characteristics
    """