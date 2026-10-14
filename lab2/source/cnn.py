import numpy as np

from layers import (
    Conv2D,
    ReLU,
    MaxPool2D,
    Flatten,
    Dense
)


class SoftmaxCrossEntropy:
    """
    Softmax + Cross-Entropy.

    Используем вместе, потому что их производная
    сильно упрощается:

        dL/dz = p - y
    """

    def forward(self, logits, target):
        self.target = target

        # Защита от переполнения exp
        shifted = logits - np.max(logits)

        exp_values = np.exp(shifted)

        self.probabilities = (
            exp_values / np.sum(exp_values)
        )

        # Cross-entropy
        loss = -np.log(
            self.probabilities[target] + 1e-12
        )

        return loss

    def backward(self):
        grad = self.probabilities.copy()

        grad[self.target] -= 1

        return grad


class CNN:
    def __init__(self):

        self.conv1 = Conv2D(
            input_channels=1,
            num_filters=8,
            kernel_size=3
        )

        self.relu1 = ReLU()

        self.pool1 = MaxPool2D(
            pool_size=2,
            stride=2
        )

        self.conv2 = Conv2D(
            input_channels=8,
            num_filters=16,
            kernel_size=3
        )

        self.relu2 = ReLU()

        self.pool2 = MaxPool2D(
            pool_size=2,
            stride=2
        )

        self.flatten = Flatten()

        self.fc1 = Dense(
            input_size=400,
            output_size=64
        )

        self.relu3 = ReLU()

        self.fc2 = Dense(
            input_size=64,
            output_size=10
        )

        self.loss_function = SoftmaxCrossEntropy()

    def forward(self, x):

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)

        x = self.fc1.forward(x)
        x = self.relu3.forward(x)

        x = self.fc2.forward(x)

        return x

    def backward(self):

        grad = self.loss_function.backward()

        grad = self.fc2.backward(grad)

        grad = self.relu3.backward(grad)

        grad = self.fc1.backward(grad)

        grad = self.flatten.backward(grad)

        grad = self.pool2.backward(grad)

        grad = self.relu2.backward(grad)

        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)

        grad = self.relu1.backward(grad)

        self.conv1.backward(grad)

    def update(self, learning_rate):

        self.conv1.update(learning_rate)

        self.conv2.update(learning_rate)

        self.fc1.update(learning_rate)

        self.fc2.update(learning_rate)

    def train_one(self, x, target, learning_rate):

        logits = self.forward(x)

        loss = self.loss_function.forward(
            logits,
            target
        )

        self.backward()

        self.update(learning_rate)

        return loss

    def predict_proba(self, x):

        logits = self.forward(x)

        shifted = logits - np.max(logits)

        exp_values = np.exp(shifted)

        probabilities = (
            exp_values / np.sum(exp_values)
        )

        return probabilities

    def predict(self, x):

        probabilities = self.predict_proba(x)

        return np.argmax(probabilities)