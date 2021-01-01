from sklearn.pipeline import Pipeline
from skorch.regressor import NeuralNetRegressor

import torch
import torch.nn.functional as F

from spectroscopy.modeling.utils import ToTorch


class DenseNN(torch.nn.Module):
    def __init__(
            self,
            num_features,
            num_units=10,
            nonlin=F.relu,
    ):
        super(DenseNN, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = torch.nn.Linear(num_features, num_units)
        self.nonlin = nonlin
        self.dense1 = torch.nn.Linear(num_units, 10)
        self.output = torch.nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X


def define_model(num_features):
    model = NeuralNetRegressor(
        DenseNN(num_units=10, num_features=num_features),
        max_epochs=20,
        lr=0.01,
        # device='cuda',  # uncomment this to train with CUDA
    ) 
    return Pipeline(steps=[
        ('toTorch', ToTorch()),
        ('model', model)
    ])


