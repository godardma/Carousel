import PyQt6
from torch import from_numpy, Tensor

from datasets.get_dataset_v1 import get_dataset_v1
from neural_net_pytorch_ import NeuralNetwork
def cost_fn(prediction, true):
    return ((prediction - true) ** 2).sum()


def train_network(inputs, outputs, dataset_name: str, epochs: int):
    model = NeuralNetwork()
    X = from_numpy(inputs)
    y = from_numpy(outputs)
    model.train_model(X, y, cost_fn=cost_fn, epochs=epochs)
    model.register_to_csv(f"models/network_{dataset_name}_{epochs}.csv")

def main_train_network(version: int = 1, N: int = 1000, epochs: int = 1000):
    inputs, outputs = get_dataset_v1(N)
    dataset_name = f'v{version}_{N}'
    train_network(inputs, outputs, dataset_name, epochs)


if __name__ == '__main__':
    main_train_network()
