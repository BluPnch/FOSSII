import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class ReLU:
    """
    ReLU activation function.

    forward:
        y = max(0, x)

    backward:
        dy/dx = 1 if x > 0 else 0
    """

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)


class Flatten:
    """
    Преобразует многомерный тензор в одномерный.

    Например:

        (32, 5, 5) -> (800,)
    """

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(-1)

    def backward(self, grad):
        return grad.reshape(self.original_shape)


class Dense:
    """
    Полносвязный слой.

    y = W @ x + b
    """

    def __init__(self, input_size, output_size):
        # Небольшая случайная инициализация весов
        self.W = (
            np.random.randn(output_size, input_size)
            * np.sqrt(2.0 / input_size)
        )

        self.b = np.zeros(output_size)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x

        # W: (output_size, input_size)
        # x: (input_size,)
        # результат: (output_size,)
        return self.W @ x + self.b

    def backward(self, grad):
        """
        grad = dL/dy

        Нужно получить:
            dL/dW
            dL/db
            dL/dx
        """

        # dL/dW
        self.dW = np.outer(grad, self.x)

        # dL/db
        self.db = grad

        # dL/dx
        return self.W.T @ grad

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class MaxPool2D:
    """
    Max Pooling 2x2.

    Например:

        [1  5]
        [3  2]

    превращается в:

        [5]
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        """
        x имеет форму:

            (channels, height, width)
        """

        self.x = x

        channels, height, width = x.shape

        out_height = (
            height - self.pool_size
        ) // self.stride + 1

        out_width = (
            width - self.pool_size
        ) // self.stride + 1

        # Все окна сразу:
        # (channels, out_height, out_width, pool_size * pool_size)
        windows = sliding_window_view(
            x,
            (self.pool_size, self.pool_size),
            axis=(1, 2)
        )[:, ::self.stride, ::self.stride]

        windows = windows.reshape(
            channels, out_height, out_width, -1
        )

        # Позиция максимума внутри каждого окна
        max_index = np.argmax(windows, axis=3)

        c_idx, i_idx, j_idx = np.indices(max_index.shape)

        # Координаты максимумов во входном тензоре
        self.max_positions = (
            c_idx,
            i_idx * self.stride + max_index // self.pool_size,
            j_idx * self.stride + max_index % self.pool_size
        )

        return x[self.max_positions]

    def backward(self, grad):
        """
        Градиент проходит только через тот элемент,
        который был максимальным во время forward.
        """

        dx = np.zeros_like(self.x)

        np.add.at(dx, self.max_positions, grad)

        return dx


class AvgPool2D:
    """
    Average Pooling 2x2.

    Например:

        [1  5]
        [3  3]

    превращается в:

        [3]
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        """
        x имеет форму:

            (channels, height, width)
        """

        self.input_shape = x.shape

        # Все окна сразу:
        # (channels, out_height, out_width, pool_size, pool_size)
        windows = sliding_window_view(
            x,
            (self.pool_size, self.pool_size),
            axis=(1, 2)
        )[:, ::self.stride, ::self.stride]

        return windows.mean(axis=(3, 4))

    def backward(self, grad):
        """
        Каждый элемент окна вошёл в среднее с весом 1 / (pool_size²),
        поэтому градиент делится поровну между всеми элементами окна.

        Если окна перекрываются, элемент получает вклад от каждого
        окна, в которое попал.
        """

        dx = np.zeros(self.input_shape)

        _, out_height, out_width = grad.shape

        share = grad / (self.pool_size * self.pool_size)

        # Элемент окна (a, b) - это вход со сдвигом (a, b)
        # от левого верхнего угла окна
        for a in range(self.pool_size):
            for b in range(self.pool_size):

                dx[
                    :,
                    a:a + self.stride * (out_height - 1) + 1:self.stride,
                    b:b + self.stride * (out_width - 1) + 1:self.stride
                ] += share

        return dx


class Conv2D:
    """
    Простая двумерная свёртка.

    input:
        (input_channels, height, width)

    filters:
        (num_filters, input_channels, kernel_size, kernel_size)
    """

    def __init__(
        self,
        input_channels,
        num_filters,
        kernel_size
    ):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        scale = np.sqrt(
            2.0 / (
                input_channels
                * kernel_size
                * kernel_size
            )
        )

        self.W = (
            np.random.randn(
                num_filters,
                input_channels,
                kernel_size,
                kernel_size
            ) * scale
        )

        self.b = np.zeros(num_filters)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        x:
            (channels, height, width)

        output:
            (num_filters, new_height, new_width)

        Используем stride=1 и padding=0.
        """

        self.x = x

        # Все окна сразу:
        # (channels, new_height, new_width, kernel_size, kernel_size)
        self.windows = sliding_window_view(
            x,
            (self.kernel_size, self.kernel_size),
            axis=(1, 2)
        )

        # output[f, i, j] = sum(W[f] * window[i, j]) + b[f]
        output = np.tensordot(
            self.W,
            self.windows,
            axes=([1, 2, 3], [0, 3, 4])
        )

        return output + self.b[:, None, None]

    def backward(self, grad):
        """
        grad:
            dL/d(output)

        Возвращаем:
            dL/d(input)
        """

        dx = np.zeros_like(self.x)

        _, out_height, out_width = grad.shape

        # dL/dW[f] = sum по (i, j) от grad[f, i, j] * window[i, j]
        self.dW = np.tensordot(
            grad,
            self.windows,
            axes=([1, 2], [1, 2])
        )

        # dL/db
        self.db = grad.sum(axis=(1, 2))

        # dL/dx: каждый элемент ядра (a, b) раздаёт градиент
        # своему сдвинутому участку входа
        for a in range(self.kernel_size):
            for b in range(self.kernel_size):

                dx[
                    :,
                    a:a + out_height,
                    b:b + out_width
                ] += np.tensordot(
                    self.W[:, :, a, b],
                    grad,
                    axes=([0], [0])
                )

        return dx

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db