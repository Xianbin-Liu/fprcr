import torch
import torch.nn as nn
from dataset import (
    fpDataSet, col_fn, create_fpDataset
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--root-log', type=str, default='log')
    return parser.parse_args()


def train_one_epoch(model, train_loader, weights, criterion, optimizer, device):
    model.train()
    train_loss_per_iter = []
    for rfp, pfp, targets in tqdm(train_loader):
        rfp, pfp, targets = rfp.to(device), pfp.to(device), targets.to(device)
        outputs = model(rfp, pfp)
        loss = sum([weights[key] * criterion[key](outputs[key], targets[:, i]) for i, key in enumerate(outputs.keys())])
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_per_iter.append(loss.item())
    return float(np.mean(train_loss_per_iter))

def eval_one_epoch(model, val_loader, weights, criterion, device):
    model.eval()
    val_loss_per_iter = []
    for rfp, pfp, targets in tqdm(val_loader):
        rfp, pfp, targets = rfp.to(device), pfp.to(device), targets.to(device)
        outputs = model(rfp, pfp)
        loss = sum([weights[key] * criterion[key](outputs[key], targets[:, i]) for i, key in enumerate(outputs.keys())])
        val_loss_per_iter.append(loss.item())
    return float(np.mean(val_loss_per_iter))

def main():
    args = arg_parser()
    torch.set_num_threads(2)
    train, val, test = create_fpDataset(data_dir=args.dataPath)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=col_fn)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=col_fn) 
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=col_fn)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')


    # 初始化模型、损失函数和优化器
    model = CondNet().to(device)
    criterion = {
        'c': nn.CrossEntropyLoss(),
        's1': nn.CrossEntropyLoss(),
        's2': nn.CrossEntropyLoss(),
        'r1': nn.CrossEntropyLoss(),
        'r2': nn.CrossEntropyLoss()
    }


    weights = {
        'c': 1,
        's1': 1,
        's2': 1,
        'r1': 1,
        'r2': 1
    }

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    test_losses = []

    best_ep = -1
    best_val_loss = 1e10
    best_test_loss = 1e10

    os.makedirs(args.root_log, exist_ok=True)
    format_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    log_dir = Path(args.root_log, format_time)
    os.makedirs(log_dir, exist_ok=True)

    log_file = Path(log_dir, 'log.json')
    model_file = Path(log_dir, 'model.pth')
    best_model_file = Path(log_dir, 'best_model.pth')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, weights, criterion, optimizer, device)
        print(f'[ TRAIN ] |  Epoch {epoch+1}, Loss: {train_loss}')
        train_losses.append(train_loss)

        val_loss = eval_one_epoch(model, val_loader, weights, criterion, device)
        print(f'[ VALID ] |  Epoch {epoch+1}, Loss: {val_loss}')
        val_losses.append(val_loss)

        # test
        test_loss = eval_one_epoch(model, test_loader, weights, criterion, device)
        print(f'[ TEST ] |  Epoch {epoch+1}, Loss: {test_loss}')     
        test_losses.append(test_loss)

        # update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ep = epoch
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_file)
            print(f'[ INFO ] |  Save model at Epoch {epoch+1}')

    # save final model
    torch.save(model.state_dict(), model_file)

    # save log
    log = {
        'args': args.__dict__,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'test_loss': test_losses,
        'best': {
            'best_ep': best_ep,
            'best_val_loss': best_val_loss,
            'best_test_loss': best_test_loss
        }
    }

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=4)

if __name__ == '__main__':
    main()