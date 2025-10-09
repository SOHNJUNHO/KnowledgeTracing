import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from pathlib import Path
import torch.nn as nn

from src.sakt.preprocessor import preprocess, load_data
from src.sakt.config import get_args
from src.sakt.model import SAKT
from src.sakt.train import train, validate_epoch
from torch.optim.lr_scheduler import ReduceLROnPlateau

MODEL_FACTORIES = {
    'SAKT': SAKT
}

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    # Load data
    df = pd.read_csv(args.dataset)

    #df = preprocess(df)
    train_data, valid_data, test_data = load_data(df, args.batch_size, args.seq_len)

    skill_nums = df["skill_id"].nunique()
    #problem_nums = df["problem_id"].nunique()

    model = MODEL_FACTORIES[args.model_type](
            skill_nums, 
            args.seq_len, 
            args.embed_size, 
            args.num_heads, 
            args.dropout,
            #num_layers=args.num_layers, 
            device
        ).to(device)

    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    # Optimizer
    optimizer = AdamW(model.parameters(), 
                      lr=args.lr) #weight_decay=1e-5
    
    scheduler = ReduceLROnPlateau(optimizer, 
                                  'max', 
                                  patience=5)  # For AUC metric


    #scheduler = ReduceLROnPlateau(
    #    optimizer, 
    #    mode='max',  # Maximize AUC
    #    patience=args.patience // 2,  # Reduce LR before early stopping
    #    factor=0.5,
    #    verbose=True
    #)

    # Save path
    save_path = Path(args.save_dir) / f"best_model_sakt_{args.embed_size}d_{args.num_heads}h.pt"

    train(
        train_data, valid_data, model, optimizer,
        args.num_epochs, args.batch_size, args.grad_clip, device
    )

    best_val_auc = train(
        train_data=train_data,
        valid_data=valid_data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        grad_clip=args.grad_clip,
        print_every=args.print_every,
        patience=args.patience,
        save_path=str(save_path),
        use_amp=args.use_amp
    )

    # Load best model for testing
    print("Evaluating on test set...")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_auc = validate_epoch(model, test_data, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()