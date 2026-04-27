import csv
from collections import Counter
from pathlib import Path

import pytest

from ai_tutor.bkt.config import BKTConfig


def test_block_size_value():
    assert BKTConfig().block_size == 818


@pytest.mark.skipif(
    not Path("data/icecream_8th_processed.csv").exists(),
    reason="dataset file is gitignored — run locally only",
)
def test_dataset_schema_and_block_size_match_processed_csv():
    csv_path = Path("data/icecream_8th_processed.csv")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        counts = Counter()
        skill_ids = set()

        for row in reader:
            counts[row[0]] += 1
            skill_ids.add(int(row[1]))

    assert header == ["﻿user_id", "skill_id", "correct", "Timestamp", "kc_id"]
    assert len(skill_ids) == 137
    assert max(skill_ids) == 137
    assert max(counts.values()) == 818
    assert BKTConfig().block_size == 818
