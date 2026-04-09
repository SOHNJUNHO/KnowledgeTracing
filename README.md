# Knowledge Tracing

A simple knowledge tracing training pipeline built with PyTorch Lightning.

This repo is now scoped to the static neural BKT package only. The previous tutor-side agent and graph integration code has been removed from this branch.

## Project Layout
```text
knowledge-tracing/
  artifacts/
    checkpoints/
    lightning_logs/
  data/
    raw/
    processed/
      assistment2009_processed.csv
  scripts/
    train_bkt.sh
  src/
    knowledge_tracing/
      __init__.py
      main.py
      train.py
      data/
        __init__.py
        datasets.py
        validation.py
      models/
        static_neural_bkt.py
  tests/
    test_bkt_data.py
    test_bkt_model.py
```

## Model

The model is a static neural-parameterized BKT variant:

`skill_id -> embedding -> 3-layer MLP -> BKT params -> Bayesian update -> predicted correctness`

It uses:
- one embedding per skill
- one static parameter set per skill
- explicit BKT recursion
- OptimNN-Reg-style soft penalties with defaults:
  - `lambda_consistency = 0.25`
  - `lambda_guess = 0.25`
  - `lambda_slip = 0.25`

## Data format

Input CSV must include:
- `user_id`
- `skill_id`
- `correct`

The preprocessing pipeline:
- validates required columns and binary correctness
- groups by `user_id`
- preserves within-user order
- truncates sequences to `block_size`
- pads with `-1000`
- creates:
  - `obs = padded[:, :-1, :]`
  - `output = padded[:, 1:, :]`

## Setup
```bash
uv sync --extra dev
```

If you want to run the shell helper directly:
```bash
bash scripts/train_bkt.sh data/processed/assistment2009_processed.csv
```

## Running Training
From the repo root:
```bash
uv run python -m knowledge_tracing.main \
  --data_path data/processed/assistment2009_processed.csv \
  --max_epochs 50
```

Checkpoints and logs will be written to `artifacts/checkpoints` and `artifacts/lightning_logs`.

## Key Files
- `src/knowledge_tracing/main.py`: CLI entrypoint.
- `src/knowledge_tracing/train.py`: Lightning training loop.
- `src/knowledge_tracing/data/datasets.py`: dataset loader and collate module.
- `src/knowledge_tracing/data/validation.py`: dataframe validation.
- `src/knowledge_tracing/models/static_neural_bkt.py`: model definition.

## Notes
- This branch intentionally excludes the old tutor integration.
- The old Transformer checkpoint has been removed from this branch.
- `data/` is kept in the repo layout and is not ignored.
