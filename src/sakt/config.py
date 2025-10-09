# src/config.py

import argparse
from pathlib import Path

def get_args():
    """
    Parses command-line arguments and returns them.
    """
    parser = argparse.ArgumentParser(description='Train Knowledge Tracing.')

    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')

    parser.add_argument('--model_type', type=str, default='SAKT', choices=['SAKT'])

    parser.add_argument('--seq_len', type=int, default=100)

    parser.add_argument('--embed_size', type=int, default=150) #256

    parser.add_argument('--num_heads', type=int, default=5) #8

    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--grad_clip', type=float, default=10)

    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of transformer blocks')

    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for training')
    
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print training status every N batches')
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()

    return args