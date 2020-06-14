import os, sys
import mxnet as mx
import utils
from mxnet.gluon import nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
import symbol_utils


class EfficientNet(nn.HybridBlock):
    def __init__(self, width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.0, input_scale=1, se_ratio=0.25, num_classes=256):
        super(EfficientNet, self).__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        expands = [1, 6, 6, 6, 6, 6, 6]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        channels = [int(round(x*width_coeff)) for x in channels] # [int(x*width) for x in channels]
        repeats = [int(round(x*depth_coeff)) for x in repeats] # [int(x*width) for x in repeats]

        self.out = nn.HybridSequential()
        if input_scale > 1.0:
            # Here, should do interpolation to resize input_image to resolution in "bi" mode
            #self.out.add(utils.UpSampling(input_scale))
            pass
        self.out.add(nn.Conv2D(channels[0], 3, 1, padding=1, use_bias=False, in_channels=3))
        self.out.add(nn.BatchNorm(scale=True))
        for i in range(7):
            self.out.add(utils.MBBlock(channels[i], channels[i+1], repeats[i], kernel_sizes[i], strides[i], expands[i], se_ratio))
        self.out.add(utils.conv_1x1_bn(channels[7], channels[8], nn.Swish()))
                    #nn.GlobalAvgPool2D(),
                    #nn.Flatten(),
                    #nn.BatchNorm(scale=False),
                    #nn.Dropout(0.4),
                    #nn.Dense(num_classes, use_bias=False, in_units=channels[8]),
                    #nn.Dense(num_classes, use_bias=False),
                    #nn.BatchNorm(scale=False))
                    #nn.Swish())
                    # utils.conv_1x1_bn(num_classes, nn.Swish()))
        # print(self.out)

    def hybrid_forward(self, F, x):
        feature = self.out(x)
        # print(feature.shape)
        return feature


class FC(nn.HybridBlock):
    def __init__(self, num_classes):
        super(FC, self).__init__()
        self.out = nn.HybridSequential()
        self.out.add(nn.BatchNorm(scale=False),
                    nn.Dropout(0.4),
                    nn.Dense(num_classes, use_bias=False),
                    nn.BatchNorm(scale=False))
    
    def hybrid_forward(self, F, x):
        return self.out(x)


def get_symbol():
    model_name="b5"
    num_classes=config.emb_size
    print("model_name: {}  emb_size {}".format(model_name ,num_classes))
    width_coeff, depth_coeff, input_resolution, dropout_rate = utils.params_dict[model_name]
    net = EfficientNet(width_coeff, depth_coeff, dropout_rate, input_scale=1.0, num_classes=num_classes)
    data = mx.sym.Variable(name='data')
    data = (data-127.5)*0.0078125
    body = net(data)
    body = symbol_utils.get_fc1(body, config.emb_size, 'E')
    return body

