import argparse
import datetime
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn
import torch.backends.cudnn as cudnn 
import json
import os
import torch.utils
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from functools import partial
from pathlib import Path
from collections import OrderedDict

from timm.utils import ModelEma
from timm.optim.optim_factory import create_optimizer, get_parameter_groups, LayerDecayC