import numpy as np
import torch
from genesis_ai.models.forecast_models import BaselineMeanModel, GRUForecaster

def test_baseline_mean_model_predicts_mean():
    model = BaselineMeanModel()
    y = np.array([1.0, 2.0, 3.0])
    model.fit(y)
    preds = model.predict(5)
    assert np.allclose(preds, np.mean(y))

def test_gru_forward_shape():
    model = GRUForecaster(input_size=8, hidden_size=16)
    x = torch.randn(4, 10, 8)
    y = model(x)
    assert y.shape == (4, 1)
