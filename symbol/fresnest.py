# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt implemented in Gluon."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import mxnet as mx
import numpy as np
import symbol_utils
#import memonger
import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
sys.path.append(os.path.join(os.path.dirname(__file__), 'resnest'))
from resnet import ResNet, Bottleneck
from mxnet import cpu

__all__ = ['resnest50', 'resnest101',
           'resnest200', 'resnest269']

def model_resnest(units, num_classes):
    bn_mom = config.bn_mom
    workspace = config.workspace
    kwargs = {'version_se' : config.net_se,
        'version_input': config.net_input,
        'version_output': config.net_output,
        'version_unit': config.net_unit,
        'version_act': config.net_act,
        'bn_mom': bn_mom,
        'workspace': workspace,
        'memonger': config.memonger,
        }
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    version_se = kwargs.get('version_se', 1)
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)
    print(version_se, version_input, version_output, version_unit, act_type, memonger)
    data = mx.sym.Variable(name='data')
    if version_input==0:
      #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
      data = mx.sym.identity(data=data, name='id')
      data = data-127.5
      data = data*0.0078125
    elif version_input==2:
      data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    else:
      data = mx.sym.identity(data=data, name='id')
      data = data-127.5
      data = data*0.0078125

    kwargs2 = {'classes': 256} #useless

    net =  ResNet(Bottleneck, units,
                      final_drop=0.2, stem_width=64,
                      dilated=False, dilation=1, #fix
                      deep_stem=True, avg_down=True,avd=True, #fix
                      use_splat=True, dropblock_prob=0.1, #fix
                      name_prefix='resnest_', **kwargs2)
    _dtype = 'float32'
    if _dtype == 'float16':
      data = mx.sym.Cast(data, dtype=np.float16)
      net.cast('float16')
    body = net(data)
    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    mx.viz.print_summary(fc1, shape={'data':(1,3,112,112)})

    if _dtype == 'float16':
      fc1 = mx.sym.Cast(fc1, dtype=np.float32)

    return fc1

def get_symbol():
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    #num_classes = int(config.emb_size / 2)
    num_classes = int(config.emb_size)
    num_layers = config.num_layers
    if num_layers == 50:
        #units = [3, 4, 6, 3]
        units = [3, 4, 10, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 200:
        units = [3, 24, 26, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    elif num_layers == 90:
        units = [3, 6, 18, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    net = model_resnest(units = units, num_classes = num_classes)

    return net



