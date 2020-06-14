import logging
import mxnet as mx
import numpy as np
from fshufflenetv2 import get_shufflenet_v2
#from fmobilenetv3 import get_symbol
from fefficientnet import get_symbol
#from fdensenet import get_symbol

# shufflenet = get_shufflenet()
#shufflenet = get_shufflenet_v2(256)
shufflenet = get_symbol()
# shufflenet = get_mobilefacenet()

# save as symbol
sym = shufflenet
#print sym
## plot network graph
mx.viz.print_summary(sym, shape={'data':(8,3,112,112)})
#mx.viz.plot_network(sym, shape={'data':(8,3,112,112)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()
