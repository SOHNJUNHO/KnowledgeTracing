import argparse

import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

from preprocessor import load_data
from utils import compute_auc, compute_loss
from saktmodel import SAKT
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAKT.')
    #parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=150)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=200)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv('data.csv')
    
    #df = preprocess(df)
    #train_data, val_data = get_data(df, args.seq_len)

    train_data, valid_data, test_data = load_data(df, args.batch_size, args.seq_len)

    
    # Model initialization
    num_items = len(df["problem"].unique())
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAKT(num_items, args.seq_len, args.embed_size, args.num_heads, args.dropout).cuda()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Train model
    train(train_data, valid_data, model, optimizer, args.num_epochs, args.batch_size, args.seq_len, args.grad_clip)
