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


class ResidualBlock(nn.HybridBlock):
    def __init__(self, channels, in_channels=None, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.stride = stride
        self.use_se = use_se
        self.in_channels = in_channels if in_channels else channels
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(self.in_channels, 1, 1, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(self.in_channels, 3, stride, padding=1, use_bias=False)
            self.bn3 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels, 1, 1, use_bias=False)
            if stride != 1 or (self.in_channels != self.channels):
                self.conv4 = nn.Conv2D(channels, 1, stride, use_bias=False)
            if self.use_se:
                self.semodule = SEModule(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x
        out = self.bn1(x)
        out1 = F.LeakyReLU(out, act_type='prelu')
        out = self.conv1(out1)
        out = self.bn2(out)
        out = F.LeakyReLU(out, act_type='prelu')
        out = self.conv2(out)
        out = self.bn3(out)
        out = F.LeakyReLU(out, act_type='prelu')
        out = self.conv3(out)
        if self.use_se:
            out = self.semodule(out, self.channels)

        if self.stride != 1 or (self.channels != self.in_channels):
            residual = self.conv4(out1)

        out = out + residual
        return out
