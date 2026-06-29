import numpy as np


def get_dataset_v2(N: int, K: float = 1, u_bar: float = 5, R: float= 10) -> tuple[np.ndarray, np.ndarray]:
    "Sample centred on the circle "
    pass


if __name__ == '__main__':
    inputs, outputs = get_dataset_v2(10)
    print(inputs.shape, inputs)
    print(outputs.shape, outputs)
