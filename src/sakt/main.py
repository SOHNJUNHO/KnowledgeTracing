import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from src.sakt.preprocessor import preprocess, load_data
from src.sakt.config import get_args
from src.sakt.utils import compute_auc, compute_loss
import src.sakt.model as SAKT
from src.sakt.train import train
from torch.optim.lr_scheduler import ReduceLROnPlateau

MODEL_FACTORIES = {
    'SAKT': SAKT
}

def main():
    args = get_args()
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

    model = MODEL_FACTORIES[args.model_type](
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