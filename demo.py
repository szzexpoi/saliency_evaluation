import numpy as np
from evaluation import cal_score
import cv2
from glob import glob
import os

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
    soft_path = '/home/umhadmin/Desktop/VQA/att_correlation/soft_attention'
    hard_path = '/home/umhadmin/Desktop/VQA/att_correlation/human_attention'
    category = load_data(soft_path)
    #initialize score structure
    metrics=['cc','sim','kld','spearman','auc']
    class_score = dict()
    avg_score = init_metrics(metrics)

    for cur_class in category:
        class_score[cur_class]=init_metrics(metrics)
        cur_path = os.path.join(soft_path,cur_class)
        cur_img = glob(os.path.join(cur_path,'*.jpg'))
        for img in cur_img:
            #reading attention maps
            soft_map = cv2.imread(img)
            soft_map = cv2.resize(soft_map,(img_cols,img_rows),interpolation = cv2.INTER_LINEAR)
            soft_map = soft_map[:,:,0]
            soft_map = soft_map.astype('float32')
            cur_name = os.path.basename(img)[:-9]
            hard_dir = os.path.join(hard_path,cur_class,cur_name+'_hard.jpg')
            hard_map = cv2.imread(hard_dir)
            hard_map = cv2.resize(hard_map,(img_cols,img_rows),interpolation = cv2.INTER_LINEAR)
            hard_map = hard_map[:,:,0]
            hard_map = hard_map.astype('float32')
            #computing score for different metrics
            avg_score['count']+=1
            class_score[cur_class]['count']+=1
            for metric_ in metrics:
                tmp_score = cal_score(soft_map,hard_map,metric_)
                avg_score[metric_] += tmp_score
                class_score[cur_class][metric_] += tmp_score


    #compute average score
    for metric_ in metrics:
        avg_score[metric_]/=avg_score['count']
        print 'The average score for ' + metric_ + ' is %f' %avg_score[metric_]

    print '\n'
    #compute the std of score across different categories
    std = init_metrics(metrics,std=True)
    for class_ in class_score:
        for metric_ in metrics:
            class_score[class_][metric_]/=class_score[class_]['count']
            std[metric_].append(class_score[class_][metric_])

    for metric_ in metrics:
        std[metric_] = np.std(std[metric_])
        print 'The std for ' + metric_ + ' is %f' %std[metric_]

if __name__ == "__main__":
    demo()
