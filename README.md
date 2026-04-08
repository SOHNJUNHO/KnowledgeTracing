# Static Neural BKT

This branch focuses only on the BKT package under `src/bkt`.

The model is a static neural-parameterized Bayesian Knowledge Tracing architecture:

`skill_id -> embedding -> MLP -> BKT parameters -> Bayesian update -> correctness probability`

It is intended as a simpler replacement for the previous Transformer-based parameter generator while preserving explicit BKT dynamics.

## Model

The model in [`src/bkt/model.py`](./src/bkt/model.py) uses:

- one embedding per skill
- one 3-layer MLP parameter head
- one static parameter set per skill
- explicit BKT state updates over the student sequence

Outputs:

- `P(L0)` initial knowledge
- `P(T)` learn
- unused compatibility slot
- `P(G)` guess
- `P(S)` slip

The training objective is:

- binary cross-entropy on predicted correctness from the BKT equation
- plus OptimNN-Reg-style soft constraints with defaults:
  - `lambda_consistency = 0.25`
  - `lambda_guess = 0.25`
  - `lambda_slip = 0.25`

## Data format

Training data must be a CSV with:

- `user_id`
- `skill_id`
- `correct`

Preprocessing is implemented in [`src/bkt/data`](./src/bkt/data):

- validate required columns
- preserve within-user order
- group rows by `user_id`
- truncate to `block_size`
- pad with `-1000`
- create:
  - `obs = padded[:, :-1, :]`
  - `output = padded[:, 1:, :]`

## Package layout

```text
src/bkt/
  __init__.py
  config.py
  main.py
  model.py
  train.py
  data/
    __init__.py
    datasets.py
    validation.py

scripts/
  train_bkt.sh

tests/
  test_bkt_data.py
  test_bkt_model.py
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Shell entrypoint:

```bash
bash scripts/train_bkt.sh data/processed/your_dataset.csv
```

Direct Python entrypoint:

```bash
python3 -m src.bkt.main \
  --data_path data/processed/your_dataset.csv \
  --batch_size 32 \
  --max_epochs 50 \
  --lr 1e-3 \
  --n_embd 64 \
  --hidden_dim 128 \
  --lambda_consistency 0.25 \
  --lambda_guess 0.25 \
  --lambda_slip 0.25
```

Training writes:

- Lightning checkpoints to `artifacts/checkpoints/`
- CSV logs to `artifacts/lightning_logs/`
- exported model checkpoint to `artifacts/checkpoints/static-neural-bkt-best.pt`

## Test

```bash
pytest tests
```

## Notes

- This branch changes only the BKT package and its local training/testing scaffolding.
- Tutor integration is intentionally out of scope for this branch.
