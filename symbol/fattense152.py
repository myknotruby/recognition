#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : residual_attention_network.py
# @Author: Piston Yang
# @Date  : 18-9-6

from __future__ import absolute_import
from mxnet.gluon import nn
from attention_module import *
import sys
import os
import symbol_utils
import mxnet as mx
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

#def ResidualBlock(channels, in_channels, stride):

class ResidualAttentionModel_90(nn.HybridBlock):

    def __init__(self, classes=1000, **kwargs):
        super(ResidualAttentionModel_90, self).__init__(**kwargs)
        """
        input size 112
        :param classes: Output classes 
        :param kwargs: 
        """
        with self.name_scope():
            n = 1
            self.n = n
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(64/n, kernel_size=3, strides=1, padding=1, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation("relu") )
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.residual_block1 = ResidualBlock(channels=64/n, in_channels=64/n, stride=1)
            self.attention_module1 = AttentionModule_stage1(64/n)
            self.attention_module1_2 = AttentionModule_stage1(64/n)
            self.attention_module1_3 = AttentionModule_stage1(64/n)
            self.residual_block2 = ResidualBlock(channels=128/n, in_channels=64/n, stride=2)
            self.attention_module2 = AttentionModule_stage2(128/n)
            self.attention_module2_2 = AttentionModule_stage2(128/n)
            self.attention_module2_3 = AttentionModule_stage2(128/n)
            self.attention_module2_4 = AttentionModule_stage2(128/n)
            self.attention_module2_5 = AttentionModule_stage2(128/n)
            self.attention_module2_6 = AttentionModule_stage2(128/n)
            self.residual_block3 = ResidualBlock(channels=256/n, in_channels=128/n, stride=2)
            self.attention_module3 = AttentionModule_stage3(256/n)
            self.attention_module3_2 = AttentionModule_stage3(256/n)
            self.residual_block4 = ResidualBlock(channels=512/n, in_channels=256/n, stride=2)
            self.residual_block5 = ResidualBlock(channels=512/n, in_channels=256/n, stride=1)
            self.residual_block6 = ResidualBlock(channels=512/n, in_channels=256/n, stride=1)
            self.conv2 = nn.HybridSequential()
            with self.conv2.name_scope():
                self.conv2.add(nn.Conv2D(512, kernel_size=1, strides=1, padding=0, use_bias=False))
                self.conv2.add(nn.BatchNorm())
                self.conv2.add(nn.Activation("relu") )



    #def hybrid_forward(self, F, x, *args, **kwargs):  #deploy
    #    x = self.conv1(x)
    #    x = self.mpool1(x)
    #    x = self.residual_block1(x)
    #    x = self.attention_module1(x)
    #    #x = self.attention_module1_2(x)
    #    #x = self.attention_module1_3(x)
    #    x = self.residual_block2(x)
    #    x = self.attention_module2(x)
    #    x = self.attention_module2_2(x)
    #    x = self.attention_module2_3(x)
    #    #x = self.attention_module2_4(x)
    #    #x = self.attention_module2_5(x)
    #    #x = self.attention_module2_6(x)
    #    x = self.residual_block3(x)
    #    x = self.attention_module3(x)
    #    x = self.attention_module3_2(x)
    #    x = self.residual_block4(x)
    #    x = self.residual_block5(x)
    #    #x = self.residual_block6(x)
    #    if self.n==2:
    #        x = self.conv2(x)


    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.attention_module2_2(x)
        x = self.attention_module2_3(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        x = self.attention_module3_2(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        if self.n==2:
            x = self.conv2(x)

        return x


def get_symbol():
    num_classes = config.emb_size
    net = ResidualAttentionModel_90(num_classes)
    data = mx.sym.Variable(name='data')
    data = data-127.5
    data = data*0.0078125

    _dtype = 'float32'
    if _dtype == 'float16':
      data = mx.sym.Cast(data, dtype=np.float16)
      net.cast('float16')
    body = net(data)
    if config.net_output == "EDUL":
        fc1, fc2, embeding = symbol_utils.get_fc1(body, config.emb_size, config.net_output)
        #mx.viz.print_summary(fc1, shape={'data':(1,3,112,112)})
        if _dtype == 'float16':
          fc1 = mx.sym.Cast(fc1, dtype=np.float32)
          fc2 = mx.sym.Cast(fc2, dtype=np.float32)
          embeding = mx.sym.Cast(embeding, dtype=np.float32)
        return fc1,fc2,embeding
    else:
        fc1 = symbol_utils.get_fc1(body, config.emb_size, config.net_output)
        if _dtype == 'float16':
          fc1 = mx.sym.Cast(fc1, dtype=np.float32)
        return fc1

