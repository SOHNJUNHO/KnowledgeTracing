import pandas as pd

from knowledge_tracing.data.datasets import create_sequences, get_data_loaders


def test_create_sequences_preserves_user_order():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "skill_id": [5, 3, 8, 2],
            "correct": [0, 1, 1, 0],
        }
    )

    sequences, group_keys = create_sequences(df, block_size=10)

    assert group_keys == [1, 2]
    assert sequences[0].tolist() == [[5.0, 0.0], [3.0, 1.0]]
    assert sequences[1].tolist() == [[8.0, 1.0], [2.0, 0.0]]


def test_get_data_loaders_smoke_case():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "skill_id": [1, 2, 2, 3, 3, 4],
            "correct": [0, 1, 1, 0, 1, 0],
        }
    )

    train_loader, val_loader, test_loader = get_data_loaders(
        df,
        batch_size=2,
        block_size=10,
        train_split=0.34,
        val_split=0.33,
        test_split=0.33,
        seed=7,
    )

    output, keys = next(iter(train_loader))
    assert len(train_loader.dataset) == 1
    assert len(val_loader.dataset) == 0
    assert len(test_loader.dataset) == 2
    assert output.shape[-1] == 2
    assert len(keys) == 1
