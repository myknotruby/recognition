#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : lb_basic_layer.py
# @Author: Piston Yang
# @Date  : 18-9-5

from mxnet.gluon import nn

class SEModule(nn.HybridBlock):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.channels = channels
        self.se = nn.HybridSequential()
        with self.name_scope():
            self.se.add( nn.GlobalAvgPool2D() )
            self.se.add( nn.Conv2D(channels // reduction,kernel_size=1, padding=0, use_bias=False) )
            self.se.add( nn.Activation("relu") )
            self.se.add( nn.Conv2D(channels, kernel_size=1, padding=0, use_bias=False) )
            self.se.add( nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        w = self.se(x)
        x = F.broadcast_mul(x, w)
        return x


#acorrding to “attention_network_def.py“
class ResidualBlock(nn.HybridBlock):
    def __init__(self, channels, in_channels=None, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.stride = stride
        self.use_se = use_se
        self.in_channels = in_channels if in_channels else channels
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(self.channels, 3, 1, padding=1, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(self.channels, 3, stride, padding=1, use_bias=False)
            self.bn3 = nn.BatchNorm()
            if self.use_se:
                self.semodule = SEModule(channels)

            if self.in_channels == self.channels:
                if self.stride == 1:
                    self.identity = 1
                else:
                    self.conv4 = nn.MaxPool2D(pool_size=1, strides=2, padding=0) 
            else:
                self.conv4 = nn.Conv2D(channels, 1, stride, use_bias=False)
                self.bn4 = nn.BatchNorm()

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.LeakyReLU(out, act_type='prelu')
        out = self.conv2(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.semodule(out, self.channels)

        if self.in_channels == self.channels:
            if self.stride != 1:
                residual = self.conv4(x) 
        else:
            residual = self.conv4(x)
            residual = self.bn4(residual)

        out = out + residual
        return out
