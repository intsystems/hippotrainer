import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def setup_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=4,
        shuffle=False,
    )
    return train_loader, val_loader


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionLoss(nn.Module):
    def __init__(self, model, lambd):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model = model
        self.lambd = lambd

    def forward(self, outputs, labels):
        ce_loss = self.cross_entropy(outputs, labels)
        l2_reg = sum(param.clone().square().sum() for param in self.model.parameters())
        loss = ce_loss + self.lambd * l2_reg
        return loss


def setup_model_and_optimizer(train_loader, val_loader):
    model = LogisticRegressionModel(4, 3)
    theta = torch.tensor([0.0], requires_grad=True)
    lambd = theta.exp()
    criterion = LogisticRegressionLoss(model, lambd)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, theta, lambd, criterion, optimizer
