from torch import from_numpy, Tensor

from datasets.get_dataset_v1 import get_dataset_v1
from neural_net_pytorch_ import NeuralNetwork


def cost_fn(prediction, true):
    return ((prediction - true) ** 2).sum()


def train_network(inputs, outputs):
    model = NeuralNetwork()
    X = from_numpy(inputs)
    y = from_numpy(outputs)
    model.train_model(X, y, cost_fn=cost_fn, epochs=10)
    model.register_to_csv("network_v1_1000,_10.csv")

def main_train_network():
    print('')
    inputs, outputs = get_dataset_v1(1000)
    print('here')
    train_network(inputs, outputs)


if __name__ == '__main__':
    main_train_network()
