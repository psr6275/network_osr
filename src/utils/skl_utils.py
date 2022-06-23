import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def set_binary_label(df, label_col, default_class='BENIGN', return_col = False):
    df_y = df[label_col].copy()
    rep_class = {}
    lab_names = df_y.unique()
    for lab in lab_names:
        if lab == default_class:
            rep_class[lab] = 0
        else:
            rep_class[lab] = 1
    if return_col:
        return df_y.replace(rep_class)
    else:
        return df.replace({label_col:rep_class})
    
def cluster_labels_2018():
    lab_cluster = {0: ['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'],
             1: ['DoS attacks-Hulk'],
             2: ['DoS attacks-GoldenEye'],
             3: ['DoS attacks-Slowloris'],
             4: ['Bot'],
             5: ['DDoS attacks-LOIC-HTTP'],
             6: ['DDOS attack-HOIC'],
             7: ['Infilteration'],
             8: ['FTP-BruteForce'],
             9: ['SSH-Bruteforce']    
    }
    lab_name = ['Web Attack','DoS attacks-Hulk','DoS attacks-GoldenEye','DoS attacks-Slowloris','Bot','DDoS attacks-LOIC-HTTP',
                'DDOS attack-HOIC','Infilteration',
                 'FTP-BruteForce','SSH-Bruteforce']
    lab_dic = {}
    for nlab in lab_cluster:
        for lab in lab_cluster[nlab]:
            lab_dic[lab] = nlab
    print(lab_dic)
    return lab_dic, lab_name

def cluster_labels_2017():
    lab_cluster = {0: ['Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection'],
             1: ['DDoS'],
             2: ['Bot'],
             3: ['DoS slowloris'],
             4: ['DoS Slowhttptest'],
             5: ['DoS Hulk'],
             6: ['DoS GoldenEye'],
             7: ['PortScan'],
             8: ['FTP-Patator'],
                 9: ['SSH-Patator']    
    }
    lab_name = ['Web Attack','DDoS','Bot','DoS slowloris','DoS Slowhttptest','DoS Hulk','DoS GoldenEye','PortScan',
                 'FTP-Patator','SSH-Patator']
    lab_dic = {}
    for nlab in lab_cluster:
        for lab in lab_cluster[nlab]:
            lab_dic[lab] = nlab
    print(lab_dic)
    return lab_dic, lab_name
def set_multiple_label(df, label_col, label_dic, return_col = False):
    df_y = df[label_col].copy()
    lab_names = df_y.unique()
    drop_idx = pd.Index([])
#     print(label_dic.keys())
    for lab in lab_names:
        if lab in label_dic.keys():
#             print('pass', lab)
            pass
        else:
            print('drop', lab)
            drop_idx = drop_idx.append(df[df[label_col]==lab].index)  
    print("original instances: ",len(df))
    print("drop intances: ", len(drop_idx))
    df = df.drop(drop_idx)
    print("after drop: ",len(df))
#     del df_y, drop_idx    
    if return_col:
        return df_y.replace(label_dic)
    else:
        return df.replace({label_col:label_dic})
        
def apply_clf_with_scaler(clf, scaler, df_x, prob = False):
    df_x_st = scaler.transform(df_x)
    if prob:
        return clf.predict_proba(df_x_st)
    else:
        return clf.predict(df_x_st)
def score_clf_with_scaler(clf, scaler, df_x, df_y):
    df_x_st = scaler.transform(df_x)
    return clf.score(df_x_st, df_y)

import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()