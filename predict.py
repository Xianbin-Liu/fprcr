import torch
import torch.nn as nn
from dataset import (
    fpDataSet, col_fn, create_fpDataset, create_fpDataset_from_file
)
from net import CondNet
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='data/uspto_condition_split')
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--best', action='store_true')
    return parser.parse_args()


def main():
    args = arg_parser()
    torch.set_num_threads(2)
    log_dir = args.log_dir
    # train, val, test = create_fpDataset(data_dir=args.dataPath)
    test = create_fpDataset_from_file(Path(args.dataPath, 'test.csv'))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=col_fn)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')


    # 初始化模型、损失函数和优化器
    model = CondNet()

    # load model weights
    if args.best:
        model.load_state_dict(torch.load(Path(log_dir, 'best_model.pth')))
    else:
        model.load_state_dict(torch.load(Path(log_dir, 'model.pth')))

    model = model.to(device)

    # save raw output for further calculation of metrics (acc et.al)
    raw_output = {
        'c': [],
        's1': [],
        's2': [],
        'r1': [],
        'r2': []
    }
    # test
    for rfp, pfp, targets in tqdm(test_loader):
        rfp, pfp, targets = rfp.to(device), pfp.to(device), targets.to(device)
        outputs = model(rfp, pfp)
        for key in outputs.keys():
            raw_output[key].append(outputs[key].detach().cpu())
    
    for key in raw_output.keys():
        raw_output[key] = torch.cat(raw_output[key], dim=0)
    # save the raw output
    torch.save(raw_output, Path(log_dir, 'raw_output.pth'))

if __name__ == '__main__':
    main()