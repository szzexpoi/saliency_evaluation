import scipy
import numpy as np
from scipy.stats import spearmanr


def cal_cc_score(salMap, fixMap):
    """
    Compute CC score between two attention maps
    """
    eps = 2.2204e-16 #regularization value
    Map_1 = salMap - np.mean(salMap)
    if np.max(Map_1) > 0:
        Map_1 = Map_1 / np.std(Map_1)
    Map_2 = fixMap - np.mean(fixMap)
    if np.max(Map_2) > 0:
        Map_2 = Map_2 / np.std(Map_2)
    if np.sum(Map_1)==0:
        Map_1+=eps
    if np.sum(Map_2)==0:
        Map_2+=eps

    score = np.corrcoef(Map_1.reshape(-1), Map_2.reshape(-1))[0][1]

    return score if not np.isnan(score) else 0

def cal_sim_score(salMap, fixMap):
    """
    Compute SIM score between two attention maps
    """
    if np.sum(salMap)>0:
        Map_1 = salMap/np.sum(salMap)
    else:
        Map_1=salMap
    if np.sum(fixMap)>0:
        Map_2 = fixMap/np.sum(fixMap)
    else:
        Map_2=fixMap

    sim_score = np.sum(np.minimum(Map_1,Map_2))

    return sim_score


def cal_kld_score(salMap,fixMap): #recommand salMap to be free-viewing attention
    """
    Compute KL-Divergence score between two attention maps
    """
    eps = 2.2204e-16 #regularization value
    if np.sum(salMap)>0:
        Map_1 = salMap/np.sum(salMap)
    else:
        Map_1=salMap
    if np.sum(fixMap)>0:
        Map_2 = fixMap/np.sum(fixMap)
    else:
        Map_2=fixMap

    kl_score = Map_2*np.log(eps+Map_2/(Map_1+eps))
    kl_score = np.sum(kl_score)

    return kl_score if not np.isnan(kl_score) else 1

def cal_spearman_corr(salMap,fixMap):
    rank_corr,_ = spearmanr(salMap.reshape(-1),fixMap.reshape(-1))

    return rank_corr if not np.isnan(rank_corr) else 0


def cal_auc_score(salMap, fixMap):
    """
    compute the AUC score for saliency prediction
    """
    fixMap = (fixMap*255>200).astype(int)
    if np.sum(fixMap)==0:
        return 0.5
    salShape = salMap.shape
    fixShape = fixMap.shape

    predicted = salMap.reshape(salShape[0]*salShape[1], -1,
                               order='F').flatten()
    actual = fixMap.reshape(fixShape[0]*fixShape[1], -1,
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


def cal_score(salMap,fixMap,metric):
    """
    API for evaluating saliency prediction with multiple metrics
    """
    if metric == 'cc':
        return cal_cc_score(salMap,fixMap)
    elif metric == 'sim':
        return cal_sim_score(salMap,fixMap)
    elif metric == 'kld':
        return cal_kld_score(salMap,fixMap)
    elif metric == 'auc':
        return cal_auc_score(salMap,fixMap)
    elif metric == 'spearman':
        return cal_spearman_corr(salMap,fixMap)
    else:
        assert 0, 'Invalid Metric:' + metric
