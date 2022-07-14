import argparse
import logging
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import make_dataloader, train_model, test_model

parser = argparse.ArgumentParser(description="Train FairAL trained models")
parser.add_argument(
    "--dataset",
    default="2017",
    type=str,
    help="dataset name: 2017",
)
parser.add_argument(
    "--OE",
    default=False,
    action="store_true",
    help="outlier exposure during attack type classification",
)
parser.add_argument(
    "--LS",
    default=False,
    action="store_true",
    help="label smoothing during training",
)
parser.add_argument(
    "--result-dir",
    default="../results/ids-dataset",
    type=str,
    metavar="PATH",
    help="result directory for the trained model and data",
)
parser.add_argument(
    "--batch-size",
    default=256,
    type=int,
    help="batch size for training",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="training epochs",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="gpu device information",
)
def binary_training(cicids_bn, epochs, batch_size, device, result_dir, save_model):
    from utils import BinaryNN
    Xtr_bn = cicids_bn[-1].transform(cicids_bn[0])
    Xte_bn = cicids_bn[-1].transform(cicids_bn[2])
    ytr_bn = cicids_bn[1]
    yte_bn = cicids_bn[3]

    train_bnloader = make_dataloader(Xtr_bn,ytr_bn, batch_size = batch_size, shuffle=True)
    test_bnloader = make_dataloader(Xte_bn,yte_bn, batch_size = batch_size, shuffle=False)

    n_features = Xtr_bn.shape[1]
    clf_bn = BinaryNN(n_features)
    criterion_bn = nn.BCELoss()
    optim_bn = optim.Adam(clf_bn.parameters(),lr=0.0001)    

    clf_bn = train_model(clf_bn, train_bnloader, optim_bn, device, criterion_bn, epochs, save_dir = result_dir, 
                     save_model = save_model, binary=True)
    print("test performance for ", save_model)
    print(test_model(clf_bn, test_bnloader, criterion_bn, device, 100.0, binary=True))
    
def mult_training(cicids_bn, cicids_m, epochs, batch_size, device, result_dir, save_model, LS, OE):
    
    from utils import MultNN
    
    Xtrm = cicids_m[-1].transform(cicids_m[0])
    Xtem = cicids_m[-1].transform(cicids_m[2])
    ytrm = cicids_m[1]
    ytem = cicids_m[3]

    train_mulloader = make_dataloader(Xtrm,ytrm.to_numpy().flatten(), batch_size = batch_size, shuffle=True)
    test_mulloader = make_dataloader(Xtem,ytem.to_numpy().flatten(), batch_size = batch_size, shuffle=False)

    num_class = len(np.unique(ytrm))
    n_features = Xtrm.shape[1]

    clf_mul = MultNN(n_features, n_hidden=32, num_class=num_class)
    optim_mul = optim.Adam(clf_mul.parameters(),lr=0.0001)

    if LS:
        criterion_mul = nn.CrossEntropyLoss(label_smoothing=0.01)
    else:
        criterion_mul = nn.CrossEntropyLoss()

    if OE:
        Xbtr = cicids_m[-1].transform(cicids_bn[0][cicids_bn[1]==0])        
        outlier_trloader = make_dataloader(Xbtr,np.zeros(len(Xbtr)),batch_size=batch_size, shuffle=True)
        criterion_oe = nn.KLDivLoss()

        clf_mul = train_model_with_oe_KL(clf_mul, train_mulloader, outlier_trloader, num_class, optim_mul, device, 
                                 criterion_mul, criterion_oe, 1.0, epochs, save_dir = result_dir, 
                     save_model = save_model,binary=False)
    else:
        train_model(clf_mul, train_mulloader, optim_mul, device, criterion_mul, epochs, save_dir = result_dir, 
                     save_model = save_model, binary=False)
    
    print("test performance for ", save_model)
    print(test_model(clf_mul, test_mulloader, criterion_mul, device, 100.0, binary=False))


def _run_training(args):
    from utils import cluster_labels_2017, cluster_labels_2018, load_cicids_binary_data, load_cicids_mult_data
    from utils import load_model_path

    device = torch.device(args.device)
    result_dir = args.result_dir
    batch_size = args.batch_size
    epochs = args.epochs
    LS = args.LS
    OE = args.OE

    assert args.dataset in ['2017', '2018']

    if args.dataset =="2017":
        lab_dic, lab_name = cluster_labels_2017()       
    else:
        lab_dic, lab_name = cluster_labels_2018()
    
    ooc_list = [None]+list(np.arange(len(lab_name)))

    for ooc_cols in ooc_list:
        cicids_bn = load_cicids_binary_data(args.dataset,result_dir,True, ooc_cols)
        cicids_m = load_cicids_mult_data(args.dataset,result_dir,True, ooc_cols)
        
        bn_save_model, mul_save_model = load_model_path(args.dataset, lab_name, epochs, ooc_cols, LS, OE)

        if os.path.isfile(os.path.join(result_dir, bn_save_model)):
            print("Already exists: ", bn_save_model)
        else:
            print("Train model to: ", bn_save_model)
            binary_training(cicids_bn, epochs, batch_size, device, result_dir, bn_save_model)
        
        if os.path.isfile(os.path.join(result_dir, mul_save_model)):
            print("Already exists: ", mul_save_model)
        else:
            print("Train model to: ", mul_save_model)
            mult_training(cicids_bn, cicids_m, epochs, batch_size, device, result_dir, mul_save_model, LS, OE)


if __name__ =="__main__":
    logger = logging.getLogger()
    args = parser.parse_args()
    _run_training(args)