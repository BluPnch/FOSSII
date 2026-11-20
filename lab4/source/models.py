import numpy as np

from layers import Conv2D, ConvTranspose2D, Linear, Rectifier, Reshape, Trainable


LATENT_DIM = 32


class Network:
    def __init__(self, **layers):
        self.layers = tuple(layers.values())
        self.trainable = {name: layer for name, layer in layers.items() if isinstance(layer, Trainable)}

    def forward(self, x, retain=False):
        for layer in self.layers:
            x = layer.forward(x, retain)
        return x

    def backward(self, gradient, *, parameter_grad=True, input_grad=False):
        for index in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[index].backward(gradient, parameter_grad, input_grad or index > 0)
        return gradient

    def parameters(self):
        for layer in self.trainable.values():
            yield from layer.parameters()

    def weights(self):
        return {
            f"{name}.{kind}": getattr(layer, kind)
            for name, layer in self.trainable.items()
            for kind in ("weight", "bias")
        }


class Encoder(Network):
    def __init__(self, rng):
        super().__init__(
            conv1=Conv2D(1, 16, rng),
            relu1=Rectifier(),
            conv2=Conv2D(16, 32, rng),
            relu2=Rectifier(),
            flatten=Reshape(-1),
            posterior=Linear(32 * 7 * 7, 2 * LATENT_DIM, rng),
        )


class Generator(Network):
    def __init__(self, rng):
        super().__init__(
            projection=Linear(LATENT_DIM, 32 * 7 * 7, rng),
            relu1=Rectifier(),
            reshape=Reshape(7, 7, 32),
            upsample1=ConvTranspose2D(32, 16, rng),
            relu2=Rectifier(),
            upsample2=ConvTranspose2D(16, 1, rng),
        )


class Discriminator(Network):
    def __init__(self, rng):
        super().__init__(
            conv1=Conv2D(1, 16, rng),
            relu1=Rectifier(0.2),
            conv2=Conv2D(16, 32, rng),
            relu2=Rectifier(0.2),
            flatten=Reshape(-1),
            classifier=Linear(32 * 7 * 7, 1, rng),
        )


class Adam:
    def __init__(self, parameters, learning_rate):
        self.parameters = tuple(parameters)
        self.learning_rate = learning_rate
        self.first = [np.zeros_like(weight) for weight, _ in self.parameters]
        self.second = [np.zeros_like(weight) for weight, _ in self.parameters]
        self.step_count = 0

    def step(self):
        beta1, beta2 = 0.9, 0.999
        self.step_count += 1
        for (weight, grad), first, second in zip(self.parameters, self.first, self.second):
            first *= beta1
            first += (1 - beta1) * grad
            second *= beta2
            second += (1 - beta2) * grad**2
            first_corrected = first / (1 - beta1**self.step_count)
            second_corrected = second / (1 - beta2**self.step_count)
            weight -= self.learning_rate * first_corrected / (np.sqrt(second_corrected) + 1e-8)
