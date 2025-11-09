import copy
import os
import sys

import numpy as np
import pandas as pd

WORKING_DIR = os.path.dirname(os.path.dirname(__file__))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

from lib.features import build_feature_pipeline


def _make_frame(length: int, *, start: int = 0) -> pd.DataFrame:
    date_ids = np.arange(start, start + length, dtype=np.int32)
    base = pd.DataFrame(
        {
            "date_id": date_ids,
            "forward_returns": np.sin(date_ids / 7.0) * 0.01,
            "risk_free_rate": np.cos(date_ids / 11.0) * 0.001,
            "market_forward_excess_returns": np.sin(date_ids / 5.0) * 0.008,
            "M1": np.linspace(0.0, 1.0, length, dtype=np.float32),
            "P1": np.linspace(1.0, 2.0, length, dtype=np.float32),
            "lagged_forward_returns": 0.0,
            "lagged_market_forward_excess_returns": 0.0,
        }
    )
    base["lagged_forward_returns"] = base["forward_returns"].shift(1).fillna(0.0)
    base["lagged_market_forward_excess_returns"] = (
        base["market_forward_excess_returns"].shift(1).fillna(0.0)
    )
    return base


def test_stateful_pipeline_streaming_matches_batch():
    train_df = _make_frame(64, start=0)
    test_df = _make_frame(45, start=10_000)

    pipeline = build_feature_pipeline(stateful=True, stateful_max_history=128)
    pipeline.fit_transform(train_df)

    reference_pipeline = copy.deepcopy(pipeline)
    streaming_pipeline = copy.deepcopy(pipeline)

    reference = reference_pipeline.transform(test_df).reset_index(drop=True)

    chunks = [
        test_df.iloc[:15],
        test_df.iloc[15:30],
        test_df.iloc[30:],
    ]
    streamed_parts = [streaming_pipeline.transform(chunk) for chunk in chunks if not chunk.empty]
    streamed = pd.concat(streamed_parts).reset_index(drop=True)

    pd.testing.assert_frame_equal(reference, streamed, check_exact=False, rtol=1e-9, atol=1e-9)
