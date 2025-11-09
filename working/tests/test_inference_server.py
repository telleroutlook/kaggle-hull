import os
import sys

import pandas as pd
import numpy as np


def test_inference_server_predict_handles_empty_batch(monkeypatch):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import inference_server as server

    class _DummyModel:
        def predict(self, X, clip=False):
            return np.zeros(len(X))

    class _DummyPipeline:
        def transform(self, df):
            return df

    def _fake_init():
        server.STATE["model"] = _DummyModel()
        server.STATE["pipeline"] = _DummyPipeline()
        server.STATE["allocation_scale"] = 20.0

    monkeypatch.setattr(server, "_ensure_model_initialized", _fake_init)
    result = server.predict(pd.DataFrame({"prediction": []}))
    assert isinstance(result, pd.DataFrame)
    assert result.empty
