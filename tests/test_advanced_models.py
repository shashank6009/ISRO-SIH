import torch
from genesis_ai.models.advanced_models import TransformerForecaster, GPRegressionModel
import gpytorch

def test_transformer_output_shape():
    model = TransformerForecaster(input_size=8, d_model=32, nhead=4)
    x = torch.randn(2, 10, 8)
    y = model(x)
    assert y.shape == (2, 1)

def test_gp_forward_pass():
    train_x = torch.linspace(0, 1, 5)
    train_y = torch.sin(train_x * 3.14)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    model.eval()
    with torch.no_grad():
        pred = model(train_x)
    assert hasattr(pred, "mean")
