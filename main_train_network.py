from torch import from_numpy

from datasets.get_dataset_v1 import get_dataset_v1
from neural_net_pytorch_ import NeuralNetwork


def cost_fn(prediction, true):
    return ((prediction - true) ** 2).sum()


def train_network(inputs, outputs, dataset_name: str, epochs: int, lr: float):
    model = NeuralNetwork()
    X = from_numpy(inputs)
    y = from_numpy(outputs)
    model.train_model(X, y, cost_fn=cost_fn, epochs=epochs, lr=lr)
    model.register_to_csv(f"models/network_{dataset_name}_{epochs}.csv")

def main_train_network(version: int = 1, N: int = 10000, epochs: int = 20000, lr: float = 0.001):
    inputs, outputs = get_dataset_v1(N, K=4, u_bar=20)
    dataset_name = f'v{version}_{N}'
    train_network(inputs, outputs, dataset_name, epochs, lr)


if __name__ == '__main__':
    main_train_network()
