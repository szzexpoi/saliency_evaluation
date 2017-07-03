import scipy
import numpy as np
from scipy.stats import spearmanr
import cv2

eps = 2.2204e-16 #regularization value

def g_filter(shape =(200,200), sigma=60):
    """
    Using Gaussian filter to generate center bias
    """
    x, y = [edge /2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in xrange(-x, x)] for j in xrange(-y, y)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter

def cal_cc_score(salMap, fixMap,center_bias):
    """
    Compute CC score between two attention maps
    """
    if np.max(salMap) > 0:
        Map_1 = np.copy(salMap)
    else:
        Map_1= salMap+eps
    if np.max(fixMap) > 0:
        Map_2 = np.copy(fixMap)
    else:
        Map_2 = fixMap + eps
    """
    salMap = salMap - np.min(salMap)
    salMap = salMap / np.max(salMap)
    Map_1 = salMap
    fixMap = fixMap - np.min(fixMap)
    fixMap = fixMap / np.max(fixMap)
    Map_2 = fixMap
    """

    #add center_bias
    if center_bias:
        cb = g_filter()
        cb = cv2.resize(cb,(Map_1.shape[1],Map_1.shape[0]),interpolation = cv2.INTER_LINEAR)
        Map_1 = Map_1*cb

    score = np.corrcoef(Map_1.reshape(-1), Map_2.reshape(-1))[0][1]
    #customize correlation coefficient
    # numerator, denominator_A, denominator_B = 0,0,0
    # mean_A, mean_B = np.mean(Map_1), np.mean(Map_2)
    # numerator = np.sum((Map_1-mean_A)*(Map_2-mean_B))
    # denominator_A = np.sum((Map_1-mean_A)**2)
    # denominator_B = np.sum((Map_2-mean_B)**2)
    #
    # score_debug = numerator/np.sqrt(denominator_A*denominator_B)

    return score if not np.isnan(score) else 0

def cal_sim_score(salMap, fixMap,center_bias):
    """
    Compute SIM score between two attention maps
    """
    #add center_bias
    if center_bias:
        cb = g_filter()
        cb = cv2.resize(cb,(salMap.shape[1],salMap.shape[0]),interpolation = cv2.INTER_LINEAR)
        salmap = salMap*cb
    else:
        salmap = np.copy(salMap)

    if np.sum(salmap)>0:
        Map_1 = salmap/np.sum(salmap)
    else:
        Map_1 = salmap+eps
    if np.sum(fixMap)>0:
        Map_2 = fixMap/np.sum(fixMap)
    else:
        Map_2 = fixMap+eps


    sim_score = np.sum(np.minimum(Map_1,Map_2))

    return sim_score


def cal_kld_score(salMap,fixMap,center_bias): #recommand salMap to be free-viewing attention
    """
    Compute KL-Divergence score between two attention maps
    """
    eps = 2.2204e-16 #regularization value

    #add center_bias
    if center_bias:
        cb = g_filter()
        cb = cv2.resize(cb,(salMap.shape[1],salMap.shape[0]),interpolation = cv2.INTER_LINEAR)
        salmap = salMap*cb
    else:
        salmap = np.copy(salMap)

    if np.sum(salmap)>0:
        Map_1 = salmap/np.sum(salmap)
    else:
        Map_1 = salmap+eps
    if np.sum(fixMap)>0:
        Map_2 = fixMap/np.sum(fixMap)
    else:
        Map_2 = fixMap+eps

    kl_score = Map_2*np.log(eps+Map_2/(Map_1+eps))
    kl_score = np.sum(kl_score)

    return kl_score if not np.isnan(kl_score) else 1

def cal_spearman_corr(salMap,fixMap,center_bias):
    #add center_bias
    if center_bias:
        cb = g_filter()
        cb = cv2.resize(cb,(salMap.shape[1],salMap.shape[0]),interpolation = cv2.INTER_LINEAR)
        salmap = salMap*cb
    else:
        salmap = np.copy(salMap)

    rank_corr,_ = spearmanr(salmap.reshape(-1),fixMap.reshape(-1))

    return rank_corr if not np.isnan(rank_corr) else 0


def cal_auc_score(salMap, fixMap,center_bias):
    """
    compute the AUC score for saliency prediction
    """
    fixmap = (fixMap*255>200).astype(int)
    if np.sum(fixmap)==0:
        return 0.5
    salShape = salMap.shape
    fixShape = fixmap.shape

    if center_bias:
        cb = g_filter()
        cb = cv2.resize(cb,(salMap.shape[1],salMap.shape[0]),interpolation = cv2.INTER_LINEAR)
        salmap = salMap*cb
    else:
        salmap = np.copy(salMap)

    predicted = salmap.reshape(salShape[0]*salShape[1], -1,
                               order='F').flatten()
    actual = fixmap.reshape(fixShape[0]*fixShape[1], -1,
                            order='F').flatten()
    labelset = np.arange(2)

    auc = area_under_curve(predicted, actual, labelset)
    return auc if not np.isnan(auc) else 0.5

def area_under_curve(predicted, actual, labelset):
    tp, fp = roc_curve(predicted, actual, np.max(labelset))
    auc = auc_from_roc(tp, fp)
    return auc

def auc_from_roc(tp, fp):
    h = np.diff(fp)
    auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
    return auc

def roc_curve(predicted, actual, cls):
    si = np.argsort(-predicted)
    tp = np.cumsum(np.single(actual[si]==cls))
    fp = np.cumsum(np.single(actual[si]!=cls))
    tp = tp/np.sum(actual==cls)
    fp = fp/np.sum(actual!=cls)
    tp = np.hstack((0.0, tp, 1.0))
    fp = np.hstack((0.0, fp, 1.0))
    return tp, fp


def cal_score(salMap,fixMap,metric,center_bias=True):
    """
    API for evaluating saliency prediction with multiple metrics
    """
    if metric == 'cc':
        return cal_cc_score(salMap,fixMap,center_bias)
    elif metric == 'sim':
        return cal_sim_score(salMap,fixMap,center_bias=False)
    elif metric == 'kld':
        return cal_kld_score(salMap,fixMap,center_bias=False)
    elif metric == 'auc':
        return cal_auc_score(salMap,fixMap,center_bias)
    elif metric == 'spearman':
        return cal_spearman_corr(salMap,fixMap,center_bias=False)
    else:
        assert 0, 'Invalid Metric:' + metric
