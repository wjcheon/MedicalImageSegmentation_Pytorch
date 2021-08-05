import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from torchensemble import VotingClassifier
from torchensemble.utils.logging import set_logger

# Define Your Base Estimator
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, data):
        data = data.view(data.size(0), -1)
        output = F.relu(self.linear1(data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output

# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train = datasets.MNIST('../Dataset', train=True, download=True, transform=transform)
test = datasets.MNIST('../Dataset', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

# Set the Logger
logger = set_logger('classification_mnist_mlp')

# Define the ensemble
model = VotingClassifier(
    estimator=MLP,
    n_estimators=10,
    cuda=True,
)

# Set the optimizer
model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

# Train and Evaluate
model.fit(
    train_loader,
    epochs=50,
    test_loader=test_loader,
)