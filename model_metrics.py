#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division, print_function
import sys
import os
import numpy as np

BIN_FILE = os.path.abspath(sys.argv[0])
BIN_ROOT = os.path.dirname(BIN_FILE)
sys.path.append(BIN_ROOT)

import pickle  # pickle模块
import joblib  # scikit-learn, it may be better to pickle
import urllib.request as request  # network

from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer  # norm='l2' 归一化

# metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

# 导入库
import torch
import torch.nn.functional as F


# 处理数据
# !pip install tensorboard
# 模型可视化


def binary_accuracy(preds, y):
    acc = None
    with torch.no_grad():
        # 四舍五入到最接近的整数
        # rounded_preds = torch.round(preds)
        # print("predictions:",preds.argmax(1))
        # print("labels:",y)
        correct = (preds.argmax(1) == y).float()
        acc = correct.sum() / len(correct)
    return acc


def binary_precison_recall_old(preds, y):
    precision = None
    recall = None
    with torch.no_grad():
        # 四舍五入到最接近的整数
        # rounded_preds = torch.round(preds)
        preds_1 = (preds.argmax(1) == 1).int()
        ground_1 = (y == 1).int()
        tp = (preds_1 == y).int()

        precision = tp.sum() / preds_1.sum()
        recall = tp.sum() / ground_1.sum()

    return precision, recall, tp.sum(), preds_1.sum(), ground_1.sum()


def binary_precison_recall(preds, y):
    precision = None
    recall = None
    with torch.no_grad():
        # 四舍五入到最接近的整数
        # rounded_preds = torch.round(preds)
        # print("preds:",preds)
        # print("yyyyyyyy:",y)
        predictions = F.softmax(preds, dim=1).squeeze()
        # print(preds)
        # print(predictions)
        y_proba, y_pred = torch.max(predictions, 1)

        # print("y_proba:",y_proba)
        # print("y_pred:",y_pred)
        # print(y)
        pred_true = y_pred == y  # TP
        pred_false = y_pred != y  # FP
        # print(pred_true,pred_false)
        # print(pred_true.int().sum(),pred_false.int().sum())

        y_pos = y == 1  # P
        y_neg = y == 0  # N
        print(y_pos.int().sum(), y_neg.int().sum())

        y_true_pos = (pred_true & y_pos).int()
        # print("y_true_pos ",y_true_pos)
        y_false_pos = (pred_false & y_neg).int()
        # print("y_false_pos ", y_false_pos)
        y_true_neg = (pred_true & y_neg).int()
        # print("y_true_neg ", y_true_neg)
        y_false_neg = (pred_false & y_pos).int()
        # print("y_false_neg ", y_false_neg)
        # print(predictions)
        # print("tp:",y_true_pos.sum(),'fp:',y_false_pos.sum(),'tn:',y_true_neg.sum(),'fn:',y_false_neg.sum())

        precision = y_true_pos.sum().float() / (y_true_pos.sum() + y_false_pos.sum())
        recall = y_true_pos.sum().float() / (y_true_pos.sum() + y_false_neg.sum())
        # TPR=TP/(TP+FN)
        tpr = recall
        # FPR=FP/(FP+TN)
        fpr = y_false_pos.sum().float() / (y_false_pos.sum() + y_true_neg.sum())
        # print("recall:",recall)
    return precision, recall, fpr, y_true_pos.sum()


def troch_pr_auc(preds, y_tmp):
    with torch.no_grad():
        predictions = F.softmax(preds, dim=1).squeeze()
        y_proba, y_pred = torch.max(predictions, 1)

        y_tmp, y_proba, y_pred = y_tmp.cpu(), y_proba.cpu(), y_pred.cpu()
        target_label_num = predictions.shape[1]
        target_names = ["label_%s" % i for i in range(target_label_num)]
        # print(y_test)
        # print(y_pred)
        print("Accuracy : %.4g" % metrics.accuracy_score(y_tmp, y_pred))
        # target_names = ['cv','backflow_cv']
        # target_names = ['click','cv','backflow_cv']
        # print("======== troch_pr_auc ====")
        # print(y_tmp,y_pred)
        print(classification_report(y_tmp, y_pred, target_names=target_names))

        C = confusion_matrix(y_tmp, y_pred)
        # [ [tn, fp], [fn, tp] ]
        print("confusion_matrix:")
        print(C)

        # AUC
        if target_label_num == 2:
            print("AUC Score : %f" % metrics.roc_auc_score(y_tmp, y_proba))
            # y_proba = model.predict_proba(X_test)[:,1]
            # print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_proba))


def binary_auc(preds, y, debug=False):
    auc_score = None
    with torch.no_grad():
        preds = torch.softmax(preds, dim=1)
        y_tmp_proba = preds[:, 1].data.cpu().numpy()
        y = y.data.cpu().numpy()
        auc_score = metrics.roc_auc_score(y, y_tmp_proba)
        # auc_score = metrics.roc_auc_score(y, y_tmp_proba, multi_class='ovo')
    return auc_score


# y_true: 1d-list-like
# y_pred: 2d-list-like
# num: 针对num个结果进行计算（num<=y_pred.shape[1]）
def precision_recall_fscore_k(y_true, y_pred, num=10):
    if not isinstance(y_pred[0], list):
        y_pred = [[each] for each in y_pred]
    #     print(y_pred)
    y_pred = [each[0:num] for each in y_pred]
    unique_label = count_unique_label(y_true, y_pred)
    # 计算每个类别的precision、recall、f1-score、support
    res = {}
    result = ''
    for each in unique_label:
        cur_res = []
        tp_fn = y_true.count(each)  # TP+FN
        # TP+FP
        tp_fp = 0
        for i in y_pred:
            if each in i:
                tp_fp += 1
        # TP
        tp = 0
        for i in range(len(y_true)):
            if y_true[i] == each and each in y_pred[i]:
                tp += 1
        support = tp_fn
        try:
            precision = round(tp / tp_fp, 2)
            recall = round(tp / tp_fn, 2)
            f1_score = round(2 / ((1 / precision) + (1 / recall)), 2)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1_score = 0
        cur_res.append(precision)
        cur_res.append(recall)
        cur_res.append(f1_score)
        cur_res.append(support)
        res[str(each)] = cur_res
    title = '\t' + 'precision@' + str(num) + '\t' + 'recall@' + str(num) + '\t' + 'f1_score@' + str(
        num) + '\t' + 'support' + '\n'
    result += title
    for k, v in sorted(res.items()):
        cur = str(k) + '\t' + str(v[0]) + '\t' + str(v[1]) + '\t' + str(v[2]) + '\t' + str(v[3]) + '\n'
        result += cur
    sums = len(y_true)
    weight_info = [(v[0] * v[3], v[1] * v[3], v[2] * v[3]) for k, v in sorted(res.items())]
    weight_precision = 0
    weight_recall = 0
    weight_f1_score = 0
    for each in weight_info:
        weight_precision += each[0]
        weight_recall += each[1]
        weight_f1_score += each[2]
    weight_precision /= sums
    weight_recall /= sums
    weight_f1_score /= sums
    last_line = 'avg_total' + '\t' + str(round(weight_precision, 2)) + '\t' + str(round(weight_recall, 2)) + '\t' + str(
        round(weight_f1_score, 2)) + '\t' + str(sums)
    result += last_line
    return round(weight_precision, 2), round(weight_recall, 2), round(weight_f1_score, 2)


# 统计所有的类别
def count_unique_label(y_true, y_pred):
    unique_label = []
    for each in y_true:
        if each not in unique_label:
            unique_label.append(each)
    for i in y_pred:
        for j in i:
            if j not in unique_label:
                unique_label.append(j)
    unique_label = list(set(unique_label))
    return unique_label


def transferPredToMat(input, n_classes):
    n_samples = len(input)
    # 生成全0的二维数组
    output = np.zeros((n_samples, n_classes)).tolist()
    for i in range(0, n_samples):
        output[i][input[i]] = 1
    return output


def auROC(y_true, y_pred):
    y_true=transferPredToMat(y_true,105)
    # print(np.array(y_pred).T[0])
    y_pred_1=transferPredToMat(np.array(y_pred).T[0],105)
    for i in range(1,20): #202020202020202020202020202020202020
        y_pred_2=transferPredToMat(np.array(y_pred).T[i],105)
        for j in range(0, len(np.array(y_pred).T[0])):
            for k in range(0, 105):
                if y_pred_2[j][k]==1:
                    y_pred_1[j][k]=1

    # print("y_pred_1 ",y_pred_1)
    # print("mmmmmmmm",y_pred)
    auroc=metrics.roc_auc_score(y_true, y_pred_1, multi_class='ovo', average='micro')
    return auroc

    # row,col = y_true.shape
    # temp = []
    # ROC = 0
    # for i in range(col):
    #     ROC = roc_auc_score(y_true[:,i], y_pred[:,i], average='micro', sample_weight=None)
    #     print("%d th AUROC: %f"%(i,ROC))
    #     temp.append(ROC)
    # for i in range(col):
    #     ROC += float(temp[i])
    # return ROC / (col + 1)

'''
class _MD(object):
    mapper = {
        str: '',
        int: 0,
        list: list,
        dict: dict,
        set: set,
        bool: False,
        float: .0
    }

    def __init__(self, obj, default=None):
        self.dict = {}
        assert obj in self.mapper, \
            'got a error type'
        self.t = obj
        if default is None:
            return
        assert isinstance(default, obj), \
            f'default ({default}) must be {obj}'
        self.v = default

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, item):
        if item not in self.dict and hasattr(self, 'v'):
            self.dict[item] = self.v
            return self.v
        elif item not in self.dict:
            if callable(self.mapper[self.t]):
                self.dict[item] = self.mapper[self.t]()
            else:
                self.dict[item] = self.mapper[self.t]
            return self.dict[item]
        return self.dict[item]


def defaultdict(obj, default=None):
    return _MD(obj, default)


def cal_precision_and_recall(pre_labels,true_labels):
    # 计算f1值
    precision = defaultdict(int, 1)
    recall = defaultdict(int, 1)
    total = defaultdict(int, 1)
    # print(pre_labels.argmax(1))
    # print(true_labels)
    for t_lab, p_lab in zip(true_labels, pre_labels):
        total[t_lab] += 1
        recall[p_lab] += 1
        print(p_lab.argmax(0))
        print(t_lab)
        # p_lab = (p_lab.argmax(1) == t_lab).float()
        # print(t_lab)
        # print(p_lab)
        if t_lab == p_lab.argmax(0):
            precision[t_lab] += 1

    for sub in precision.dict:
        pre = precision[sub] / recall[sub]
        rec = precision[sub] / total[sub]
        F1 = (2 * pre * rec) / (pre + rec)
        print(f"{str(sub)}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")


# if __name__=="__main__":
#     y_true = [1, 2, 3]
#     y_pred = [1, 1, 3]
#     cal_precision_and_recall(y_true,y_pred)
'''
