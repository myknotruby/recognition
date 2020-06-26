#!/usr/bin/env python
# encoding: utf-8
"""
PyRoc.py
1) read same_score.txt, diff_score.txt
2) calculate roc rate[0.01, 0.001, 0.0001, 0.00001]
3) calculate err
last edited 2015.5.15
"""
import sys,os
import random
import math
import numpy as np


def roc_accuracy(same_score, diff_score, thresholds=[]):
    """"when The number of positive and negative are basic equivalent, as a reference value """
    dist = same_score + diff_score
    actual_issame = np.zeros(len(dist), dtype=np.bool)
    actual_issame[:len(same_score)] = True
    dist = np.array( dist )

    deviation = float(len(same_score))/dist.size
    if deviation<0.3:
        print( "the pos-neg pairs are deviation (accuracy value is useless)! %0.4f "%(deviation)  )

    accuracy = []
    for threshold in thresholds:
        threshold = np.array(threshold, dtype=float)
        predict_issame = np.greater(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
        tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/dist.size
        accuracy.append( acc )
    return ['%0.4f'%x for x in accuracy]


def roc_curve( same_score, diff_score ):
        if __name__ != '__main__':
	        print "PyRoC ..."	
        same_length = len( same_score )
        diff_length = len( diff_score )

        sorted_diff = sorted(diff_score)
        idx = []
        threshold = []
        far = []
        rate = [100,1000,10000,100000,1000000,10000000]
        for i,r in enumerate(rate):
                idx.append( int(diff_length*(1-1.0/r)) )
                threshold.append( sorted_diff[int(idx[i])] )
                new_list = [n for n in same_score if n<threshold[i]]
                far.append( 1.0 - float(len(new_list))/same_length ) #wrong naming, TAR
        far = [ ('%.4f'%v) for v in far ]
        threshold = [ ('%.4f'%v) for v in threshold ]
        #accuracy = roc_accuracy( same_score, diff_score, threshold )
        #return far, threshold, accuracy
        return far, threshold


def roc_err( same_score, diff_score ):
    """faster"""
    sorted_same = sorted( same_score, reverse=True ) 
    sorted_diff = sorted( diff_score, reverse=True ) #decent
    P = len( same_score )
    N = len( diff_score )
    
    fp = 0.0  
    tp = 0.0 
    min_val = 10000.0 
    intersection = ()
    idx_diff = 0
    idx_same = 0
    over = False
    while idx_same<P:
        if over:
            break
        while 1:
            if idx_diff < N:
                curve_elem = (fp/N, tp/P, sorted_diff[idx_diff])
                cur_score = abs(curve_elem[0]+curve_elem[1] - 1.0)
                if cur_score <= min_val:
                    intersection = curve_elem;
                    min_val = cur_score;
                else:
                    over = True
                    break
                
                if sorted_diff[idx_diff] < sorted_same[idx_same]:
                    tp += 1
                    idx_same += 1
                else:
                    fp += 1
                    idx_diff += 1
                    break
            else:
                over = True
                break
    # err = intersection
    #return '%.4f %.4f'%(intersection[1], intersection[0] )
    return intersection[1], intersection[0]

def open_file( same_txt, diff_txt ):
    same_score = []
    diff_score = []
    with open(same_txt,'r') as same:
            lines = same.readlines()
            for line in lines:
                    same_score.append( float(line) )
    with open(diff_txt,'r') as diff:
            lines = diff.readlines()
            for line in lines:
                    diff_score.append( float(line) )
    return same_score,diff_score



#===========================================

def open_combinefile( score_label_txt):
    """score_label_txt format: score label
       while label=1 if same or diff
    """
    same_score = []
    diff_score = []
    with open(score_label_txt,'r') as same:
            lines = same.readlines()
            for line_ in lines:
                line = line_.rsplit()
                if int(line[1])==1:
                    same_score.append( float(line[0]) )
                else:
                    diff_score.append( float(line[0]) )
    return same_score,diff_score


def load_pair_ground_truth(pair_ground_truth_path):
    file = open(pair_ground_truth_path)
    records = file.readlines()
    print len(records)
    dict = {}
    for record in records:
        splited = record.rstrip().split(' ')
        if len(splited) != 2:
            break
        dict[splited[0]] = splited[1]
    file.close()

    return dict





def open_score( ab_score, pair_ground_truth_path=None, savescore=False ):
    """ab_score format: nameA nameB scoreAB"""
    def get_label(t1, t2):
        #print int(t1), int(t2),int(t1)==int(t2)
        #if int(t1)==int(t2): #jiangsu
        if t1==t2:
            return 1
        else:
            return 0

    def save( score , scoretxt ):
        with open(scoretxt, 'w') as f:
            for s in score:
                f.write('%f\n'%(s))
    def save_sl( score , scoretxt ):
        with open(scoretxt, 'w') as f:
            for s in score:
                f.write('%f %d\n'%(s[0],s[1]))

    pair_dict = None
    if pair_ground_truth_path is not None:
        pair_dict = load_pair_ground_truth( pair_ground_truth_path )
        print len(pair_dict)

    score_label = []
    same_score = []
    diff_score = []
    with open(ab_score,'r') as txt:
            lines = txt.readlines()
            for line_ in lines:
                line = line_.rsplit(',')
                if len(line)==3:
                    if pair_dict is not None:
                        #'format=(md1,md2,score)'
                        #import pdb; pdb.set_trace()
                        if get_label( pair_dict[line[0]].rsplit('/')[0], pair_dict[line[1]].rsplit('/')[0]): #md5-verification
                            same_score.append( float(line[2]) )
                            #score_label.append( [float(line[2]),1] )
                        else:
                            diff_score.append( float(line[2]) )
                    else:
                        #'format=(dir1/img1,dir2/img2,score)'
                        #if get_label(os.path.basename(line[0]).rsplit('.')[0], os.path.basename(line[1]).rsplit('.')[0]): #jiangsu
                        if get_label(os.path.basename(line[0]).rsplit('/')[0], os.path.basename(line[1]).rsplit('/')[0]): #verification
                            same_score.append( float(line[2]) )
                            #score_label.append( [float(line[2]),1] )
                        else:
                            diff_score.append( float(line[2]) )
                            #score_label.append( [float(line[2]),0] )
                elif len(line)==4:
                    #'format=(img1,img2,score,label)'
                    if int(line[3])==1:
                        same_score.append( float(line[2]) )
                        #score_label.append( [float(line[2]),1] )
                    else:
                        diff_score.append( float(line[2]) )
                        #score_label.append( [float(line[2]),0] )
    print len(same_score), len(diff_score)
    if savescore:
        save(same_score, 'same_fc.txt')
        save(diff_score, 'diff_fc.txt')
        save_sl(score_label, 'output_score.txt')
    return same_score,diff_score



        
def main(argv):
    #same_score, diff_score = open_combinefile( 'output_score.txt')
    if len(argv)==2:
        print 'procing ', argv[1]
        same_score, diff_score = open_score( argv[1])
        print "calculating ..."
        print roc_curve( same_score, diff_score )
        print roc_err( same_score, diff_score ); 
        return 0
    elif len(argv)==3:
        print 'procing ', argv[1], '  ground_truth ', argv[2]
        same_score, diff_score = open_score( argv[1], argv[2])
        print "calculating ..."
        print roc_curve( same_score, diff_score )
        print roc_err( same_score, diff_score );
    else:
        print 'usage: python pyroc.py scores.txt [ground_truth.txt]'


		
if __name__ == '__main__':
	print "PyRoC - ROC Curve Generator"	
	main(sys.argv)	
