import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from preprocessor import preprocess, load_data
from utils import compute_auc, compute_loss
import SAKT.models as models
from train import train
from torch.optim.lr_scheduler import ReduceLROnPlateau


#if __name__ == "__main__":
def main():
    parser = argparse.ArgumentParser(description='Train Knowledge Tracing.')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='SAKT', choices=['SAKT'])
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
    df = pd.read_csv(args.dataset)

    #df = preprocess(df)
    train_data, valid_data, test_data = load_data(df, args.batch_size, args.seq_len)

    skill_nums = df["skill_id"].nunique()
    problem_nums = df["problem_id"].nunique()

    device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
    )

    if args.model_type == 'SAKT':
        model = models(
            skill_nums, 
            args.seq_len, 
            args.embed_size, 
            args.num_heads, 
            args.dropout, 
            device
        ).to(device)

        # Optimizer
        optimizer = Adam(model.parameters(), lr=args.lr) #weight_decay=1e-5
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)  # For AUC metric

        train(
            train_data, valid_data, model, optimizer,
            args.num_epochs, args.batch_size, args.grad_clip, device
        )


if __name__ == "__main__":
    main()