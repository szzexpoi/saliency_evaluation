import numpy as np
from evaluation import cal_score
import cv2
from glob import glob
import os
from os import listdir
from os.path import isfile, join
import sys

def init_metrics(metrics,std=False):
    """
    initializing dictionary for storing results
    """
    metric = dict()
    for k in metrics:
        if std == False:
            metric[k] = 0
        else:
            metric[k] = []
    if std == False:
        metric['count'] = 0

    return metric

def load_data(path):
    category = glob(path+'/*/')
    target = [] #storing the class label for future analysis
    flname = []
    for cur in category:
        target.append(cur[len(path)+1:-1])
    return target

def demo():
    img_rows, img_cols = 448, 448
    #reading data
    soft_path = '/home/luoyan/project/lua/grad-cam/output'
    hard_path = '/home/luoyan/project/matlab/saliency_torch/data/salicon_mouse_val_salmap'
    category = load_data(soft_path)
    #initialize score structure
    metrics=['cc','sim','kld','spearman','auc']
    class_score = dict()
    avg_score = init_metrics(metrics)

    files = [f for f in listdir(hard_path) if isfile(join(hard_path,f)) and f.endswith('.jpg')]
    for idx, file in enumerate(files):
        print('process {}-th/{} {}'.format(idx+1,len(files), file))
        soft_map = cv2.imread(join(soft_path,file))
        soft_map = cv2.resize(soft_map,(img_cols,img_rows),interpolation = cv2.INTER_LINEAR)
        soft_map = np.mean(soft_map,axis=2)
        soft_map = soft_map.astype('float32')
        hard_map = cv2.imread(join(hard_path,file))
        hard_map = cv2.resize(hard_map,(img_cols,img_rows),interpolation = cv2.INTER_LINEAR)
        hard_map = np.mean(hard_map,axis=2)
        hard_map = hard_map.astype('float32')

        avg_score['count']+=1
        for metric_ in metrics:
            tmp_score = cal_score(soft_map,hard_map,metric_)
            avg_score[metric_] += tmp_score

    #compute average score
    for metric_ in metrics:
        avg_score[metric_]/=avg_score['count']
        print 'The average score for ' + metric_ + ' is %1.4f' %avg_score[metric_]
    
if __name__ == "__main__":
    demo()
