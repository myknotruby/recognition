import os, sys
import mxnet as mx
import numpy as np
import symbol_utils
import math
#from mxnet.base import _Null
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

withBN = True
bn_mom = 0.9
def conv_branch_pool(data, units, filters, workspace):
  body = data
  #body = mx.sym.Convolution(data=body, no_bias=False, num_filter=32, kernel=(5, 5), stride=(2,2), pad=(2, 2),
  body = mx.sym.Convolution(data=body, no_bias=False, num_filter=24, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                            name= "conv%d_%d"%(0, 1), workspace=workspace)
  if withBN:
      bn = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (0, 1)))
      body = mx.sym.LeakyReLU(data=bn, act_type='leaky', name="relu%d_%d" % (0, 1))
  else:
      body = mx.sym.LeakyReLU(data = body, act_type='prelu', name = "relu%d_%d" % (0, 1))
  for i in xrange(len(units)):
    f = filters[i]
    idx = 1
    if i>=1:
        _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, idx), lr_mult=1.0)
        _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, idx), lr_mult=2.0, wd_mult=0.0)
        body1 = mx.sym.Convolution(data=body, weight = _weight, bias = _bias, num_filter=f, kernel=(3, 3), stride=(2,2), pad=(1, 1),
                                  name= "conv%d_%d"%(i+1, idx), workspace=workspace)
        if withBN:
            bn = mx.sym.BatchNorm(data=body1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (i+1, idx)))
            #body1 = mx.sym.LeakyReLU(data=bn, act_type='leaky', name="relu%d_%d" % (i+1, idx))
            body1  = mx.sym.Activation(data=bn, act_type='relu', name="relu%d_%d" % (i+1, idx))
        else:
            body1 = mx.sym.LeakyReLU(data = body1, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
        idx += 1

        _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, idx), lr_mult=1.0)
        _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, idx), lr_mult=2.0, wd_mult=0.0)
        body1 = mx.sym.Convolution(data=body1, weight=_weight, bias=_bias, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                    name= "conv%d_%d"%(i+1, idx), workspace=workspace)
        if withBN:
            body1 = mx.sym.BatchNorm(data=body1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (i+1, idx)))
            ##body1 = mx.sym.LeakyReLU(data=bn, act_type='relu', name="relu%d_%d" % (i+1, idx))
        else:
            body1 = mx.sym.LeakyReLU(data = body1, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
        idx += 1

        body = mx.sym.Pooling(data = body, global_pool=False, kernel=(2,2), stride=(2,2),pool_type='avg', pooling_convention='full',name='pool%d_%d'%(i+1, idx))
        idx += 1
        _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, idx), lr_mult=1.0)
        _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, idx), lr_mult=2.0, wd_mult=0.0)
        body = mx.sym.Convolution(data=body, weight=_weight, bias=_bias, num_filter=f, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                    name= "conv%d_%d"%(i+1, idx), workspace=workspace)
        if withBN:
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (i+1, idx)))
            ##body = mx.sym.LeakyReLU(data=bn, act_type='relu', name="relu%d_%d" % (i+1, idx))
        else:
            body = mx.sym.LeakyReLU(data = body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
        body = body1 + body
        idx += 1
    for j in xrange(units[i]):
      _body = mx.sym.Convolution(data=body, no_bias=True, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)
      if withBN:
          bn = mx.sym.BatchNorm(data=_body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (i+1, idx)))
          #_body = mx.sym.LeakyReLU(data=bn, act_type='leaky', name="relu%d_%d" % (i+1, idx))
          _body  = mx.sym.Activation(data=bn, act_type='relu', name="relu%d_%d" % (i+1, idx))
      else:
          _body = mx.sym.LeakyReLU(data = _body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
      idx+=1

      _body = mx.sym.Convolution(data=_body, no_bias=True, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)
      if withBN:
          _body = mx.sym.BatchNorm(data=_body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (i+1, idx)))
          ##_body = mx.sym.LeakyReLU(data=bn, act_type='relu', name="relu%d_%d" % (i+1, idx))
      else:
          _body = mx.sym.LeakyReLU(data = _body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
      idx+=1
      body = body+_body

  node = len(units)+1
  body = mx.sym.Convolution(data=body, no_bias=False, num_filter=512, kernel=(1, 1), stride=(1,1), pad=(0, 0),
                            name= "conv%d_%d"%(node, 1), workspace=workspace)
  if withBN:
      bn = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=("bn%d_%d" % (node, idx)))
      #body = mx.sym.LeakyReLU(data=bn, act_type='leaky', name="relu%d_%d" % (node, idx))
      body  = mx.sym.Activation(data=bn, act_type='relu', name="relu%d_%d" % (node, idx))
  else:
      body = mx.sym.LeakyReLU(data = body, act_type='prelu', name = "relu%d_%d" % (node, idx))

  #import pdb; pdb.set_trace()
  return body



def conv_main(data, units, filters, workspace):
  body = data
  body = mx.sym.Convolution(data=body, no_bias=True, num_filter=32, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                            name= "conv%d_%d"%(0, 1), workspace=workspace)
  body = mx.sym.LeakyReLU(data = body, act_type='prelu', name = "relu%d_%d" % (0, 1))
  for i in xrange(len(units)):
    f = filters[i]
    idx = 1
    _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, idx), lr_mult=1.0)
    _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, idx), lr_mult=2.0, wd_mult=0.0)
    body1 = mx.sym.Convolution(data=body, weight = _weight, bias = _bias, num_filter=f, kernel=(3, 3), stride=(2,2), pad=(1, 1),
                              name= "conv%d_%d"%(i+1, idx), workspace=workspace)
    body1 = mx.sym.LeakyReLU(data = body1, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
    idx += 1

    _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, idx), lr_mult=1.0)
    _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, idx), lr_mult=2.0, wd_mult=0.0)
    body1 = mx.sym.Convolution(data=body1, weight=_weight, bias=_bias, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)
    body1 = mx.sym.LeakyReLU(data = body1, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
    idx += 1
    body = mx.sym.Pooling(data = body, global_pool=False, kernel=(2,2), stride=(2,2),pool_type='avg', pooling_convention='full', name='pool%d_%d'%(i+1, idx))
    idx += 1
    _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, idx), lr_mult=1.0)
    _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, idx), lr_mult=2.0, wd_mult=0.0)
    body = mx.sym.Convolution(data=body, weight=_weight, bias=_bias, num_filter=f, kernel=(1, 1), stride=(1,1), pad=(0, 0),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)
    body = mx.sym.LeakyReLU(data = body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
    body = body1 + body
    idx += 1
    for j in xrange(units[i]):
      _body = mx.sym.Convolution(data=body, no_bias=True, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)

      _body = mx.sym.LeakyReLU(data = _body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
      idx+=1
      _body = mx.sym.Convolution(data=_body, no_bias=True, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)
      _body = mx.sym.LeakyReLU(data = _body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
      idx+=1
      body = body+_body

  return body

def get_symbol():
  _dtype = config.dtype
  num_classes = config.emb_size
  fc_type = 'E' #config.net_output
  workspace = config.workspace
  num_layers = config.num_layers
  if num_layers==64:
    units = [3,8,16,3]
    filters = [64,128,256,512]
  elif num_layers==20:
    units = [1,2,4,1]
    filters = [64,128,256,512]
    #filters = [64, 256, 512, 1024]
  elif num_layers==36:
    units = [2,4,8,2]
    filters = [64,128,256,512]
    #filters = [64, 256, 512, 1024]
  elif num_layers==44:
    units = [3,4,4,4,3]
    filters = [16,32,64,128,256]
  elif num_layers==60:
    units = [3,8,14,3]
    filters = [64,128,256,512]
  elif num_layers==68:
    units = [3,4,8,12,3]
    filters = [16,32,64,128,256]
  elif num_layers==74:
    units = [3,6,8,12,3]
    filters = [24,32,64,128,256]
    #filters = [32,64,128,256,256]
  elif num_layers==104:
    units = [3,8,36,3]
    filters = [64,128,256,512]
    #filters = [64, 256, 512, 1024]
  data = mx.symbol.Variable('data')
  data = data-127.5
  data = data*0.0078125
  if _dtype == 'float16':
      data = mx.sym.Cast(data, dtype=np.float16)
  body = None
  if num_layers==44 or num_layers==68 or num_layers==74:
    body = conv_branch_pool(data = data, units = units, filters = filters, workspace = workspace)
  else:
    body = conv_main(data = data, units = units, filters = filters, workspace = workspace)
  #_weight = mx.symbol.Variable("fc1_weight", lr_mult=1.0)
  #_bias = mx.symbol.Variable("fc1_bias", lr_mult=2.0, wd_mult=0.0)
  #fc1 = mx.sym.FullyConnected(data=body, weight=_weight, bias=_bias, num_hidden=num_classes, name='fc1')
  if _dtype == 'float16':
      body = mx.sym.Cast(body, dtype=np.float16)

  fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)

  if _dtype == 'float16':
      fc1 = mx.sym.Cast(fc1, dtype=np.float32)
  #fc1 = mx.sym.Pooling(data = body, global_pool=True, kernel=(2,2), stride=(2,2),pool_type='avg', pooling_convention='full', name='pool_fc1')
  return fc1
  
def init_weights(sym, data_shape_dict, num_layers):
  arg_name = sym.list_arguments()
  aux_name = sym.list_auxiliary_states()
  arg_shape, aaa, aux_shape = sym.infer_shape(**data_shape_dict)
  #print(data_shape_dict)
  #print(arg_name)
  #print(arg_shape)
  arg_params = {}
  aux_params = None
  #print(aaa)
  #print(aux_shape)
  arg_shape_dict = dict(zip(arg_name, arg_shape))
  aux_shape_dict = dict(zip(aux_name, aux_shape))
  #print(aux_shape)
  #print(aux_params)
  #print(arg_shape_dict)
  for k,v in arg_shape_dict.iteritems():
    if k.startswith('conv') and k.endswith('_weight'):
      if not k.find('_1_')>=0:
        if num_layers<100:
          arg_params[k] = mx.random.normal(0, 0.01, shape=v)
          print('init', k)
    if k.endswith('_bias'):
      arg_params[k] = mx.nd.zeros(shape=v)
      print('init', k)
  return arg_params, aux_params

