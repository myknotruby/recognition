import os
import sys
import numpy as np
import cPickle
from PIL import Image


def pinjie(x1, x2, score, issame):
  in_root = '/train/mxnet-train/aligned_data/zhongya-test'
  out_root = '/home/liaohuan/smbshare/tmp'
  img1 = Image.open('%s/%s'%(in_root,x1))
  img2 = Image.open('%s/%s'%(in_root,x2))
  width = 112
  height = 112
  result = Image.new(img1.mode, (2*width, height))
  result.paste(img1, box=(0,0))
  result.paste(img2, box=(width,0))
  if issame:
    result.save('{}/same/{}.{}'.format(out_root, score, 'jpg'))
  else:
    result.save('{}/diff/{}.{}'.format(out_root, score, 'jpg'))

def check_pkl(score_pkl):
  threshold = [0.2852]
  pkl = open('./scores.pkl','rb')
  same = cPickle.load(pkl)
  #diff = cPickle.load(pkl)
  diff = []
  #import pdb;pdb.set_trace()
  for t in threshold:
    for s in same:
        if s[2]<t:
            print(s)
            pinjie(s[0], s[1], s[2], True)
    for d in diff:
        if d[2]<t:
            print(d)
            pinjie(s[0], s[1], s[2],False)

if __name__ == '__main__':

  check_pkl(sys.argv[1])
