"""
Different types of performance functions
"""
import numpy as np


def avg_precision(prediction, targets, svm=False, size_1=101, size_2=11):
    '''
    targets : {0, 1} or {-1, 1}
    return, average precision
    see, "The PASCAL Visual Object Classes (VOC) Challenge"
    '''
    precision = np.ndarray(shape=(prediction.shape[1], 0))
    recall = np.ndarray(shape=(prediction.shape[1], 0))
    def get_pr(prediction, targets, threshold):
        '''
        precision, recall wrt a threshold
        '''
        predict = np.array(
            prediction > (np.ones_like(prediction) * thresholds), dtype=int)
        tp = np.array(
            np.logical_and(predict == targets,
                           predict == np.ones_like(predict)), dtype=int)
        fp = np.array(
            np.logical_and(predict != targets,
                           predict == np.ones_like(predict)), dtype=int)
        fn = np.array(
            np.logical_and(predict != targets,
                           predict == np.zeros_like(predict)), dtype=int)
        precision_ = (tp.sum(axis=0).astype(float) /
                      (tp.sum(axis=0) + fp.sum(axis=0)))[np.newaxis].T
        recall_ = (tp.sum(axis=0).astype(float) /
                   (tp.sum(axis=0) + fn.sum(axis=0)))[np.newaxis].T
        return precision_, recall_
    low = 0
    if svm:
        low = -1
    for threshold in np.linspace(low, 1, size_1):
        precision_, recall_,  = get_pr(prediction, targets, threshold)
        precision = np.hstack((precision, precision_))
        recall = np.hstack((recall, recall_))
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    ap = np.zeros(prediction.shape[1], dtype=np.float16)
    for t in np.linspace(0, 1, size_2):
        h = (recall >= t) * precision
        ap += h.max(axis=1)
    ap /= size_2
    return ap
