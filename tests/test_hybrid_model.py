import torch
from genesis_ai.models.hybrid_model import HybridForecastModel, HybridWithGP

def test_hybrid_forward_shape():
    model = HybridForecastModel(input_size=6, hidden_size=16)
    x = torch.randn(3, 8, 6)
    out = model(x)
    assert out.shape == (3, 1)

def test_gp_wrapper_predicts():
    torch.manual_seed(0)
    preds = torch.linspace(0, 1, 10).unsqueeze(1)
    y = torch.sin(preds * 3.14)
    gp = HybridWithGP()
    gp.fit_gp(preds, y)
    mean, var = gp.predict(preds)
    assert mean.shape == preds.squeeze().shape
    assert var.min() >= 0
