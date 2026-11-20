import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


KERNEL = 4
STRIDE = 2
PADDING = 1


def image_columns(images):
    """Ищет окна свёртки в батче изображений (N, H, W, C)."""
    padded = np.pad(images, ((0, 0), (PADDING, PADDING), (PADDING, PADDING), (0, 0)))
    windows = sliding_window_view(padded, (KERNEL, KERNEL), axis=(1, 2))
    windows = windows[:, ::STRIDE, ::STRIDE].transpose(0, 1, 2, 4, 5, 3)
    return np.ascontiguousarray(windows).reshape(-1, KERNEL**2 * images.shape[-1])


def columns_image(columns, shape):
    """Складывает вклады перекрывающихся окон в изображении."""
    batch, height, width, channels = shape
    out_height, out_width = height // STRIDE, width // STRIDE
    patches = columns.reshape(batch, out_height, out_width, KERNEL, KERNEL, channels)
    padded = np.zeros((batch, height + 2 * PADDING, width + 2 * PADDING, channels), dtype=columns.dtype)
    for row in range(KERNEL):
        for col in range(KERNEL):
            padded[:, row:row + height:STRIDE, col:col + width:STRIDE] += patches[:, :, :, row, col]
    return np.ascontiguousarray(padded[:, PADDING:-PADDING, PADDING:-PADDING])


class Trainable:
    def __init__(self, weight_shape, bias_size, fan_in, rng):
        limit = 1 / np.sqrt(fan_in)
        self.weight = rng.uniform(-limit, limit, weight_shape).astype(np.float32)
        self.bias = rng.uniform(-limit, limit, bias_size).astype(np.float32)
        self.weight_grad = np.empty_like(self.weight)
        self.bias_grad = np.empty_like(self.bias)

    def parameters(self):
        return ((self.weight, self.weight_grad), (self.bias, self.bias_grad))


class Linear(Trainable):
    def __init__(self, input_size, output_size, rng):
        super().__init__((input_size, output_size), output_size, input_size, rng)

    def forward(self, x, retain):
        if retain:
            self.input = x
        return x @ self.weight + self.bias

    def backward(self, gradient, parameter_grad, input_grad):
        if parameter_grad:
            self.weight_grad[:] = self.input.T @ gradient
            self.bias_grad[:] = gradient.sum(axis=0)
        del self.input
        if input_grad:
            return gradient @ self.weight.T


class Conv2D(Trainable):
    def __init__(self, input_channels, output_channels, rng):
        fan_in = KERNEL**2 * input_channels
        super().__init__((fan_in, output_channels), output_channels, fan_in, rng)

    def forward(self, images, retain):
        columns = image_columns(images)
        if retain:
            self.cache = columns, images.shape
        batch, height, width, _ = images.shape
        output = columns @ self.weight + self.bias
        return output.reshape(batch, height // STRIDE, width // STRIDE, -1)

    def backward(self, gradient, parameter_grad, input_grad):
        columns, shape = self.cache
        del self.cache
        gradient = gradient.reshape(-1, self.bias.size)
        if parameter_grad:
            self.weight_grad[:] = columns.T @ gradient
            self.bias_grad[:] = gradient.sum(axis=0)
        if input_grad:
            return columns_image(gradient @ self.weight.T, shape)


class ConvTranspose2D(Trainable):
    def __init__(self, input_channels, output_channels, rng):
        fan_in = KERNEL**2 * output_channels
        super().__init__((input_channels, fan_in), output_channels, fan_in, rng)

    def forward(self, images, retain):
        batch, height, width, channels = images.shape
        flat = images.reshape(-1, channels)
        if retain:
            self.cache = flat, images.shape
        shape = (batch, height * STRIDE, width * STRIDE, self.bias.size)
        output = columns_image(flat @ self.weight, shape)
        return output + self.bias

    def backward(self, gradient, parameter_grad, input_grad):
        flat, shape = self.cache
        del self.cache
        columns = image_columns(gradient)
        if parameter_grad:
            self.weight_grad[:] = flat.T @ columns
            self.bias_grad[:] = gradient.sum(axis=(0, 1, 2))
        if input_grad:
            return (columns @ self.weight.T).reshape(shape)


class Rectifier:
    def __init__(self, negative_slope=0):
        self.negative_slope = np.float32(negative_slope)

    def forward(self, x, retain):
        if retain:
            self.positive = x > 0
        if self.negative_slope:
            return np.where(x > 0, x, self.negative_slope * x)
        return np.maximum(x, 0)

    def backward(self, gradient, parameter_grad, input_grad):
        slopes = np.where(self.positive, np.float32(1), self.negative_slope)
        del self.positive
        return gradient * slopes


class Reshape:
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x, retain):
        if retain:
            self.input_shape = x.shape
        return x.reshape(len(x), *self.shape)

    def backward(self, gradient, parameter_grad, input_grad):
        return gradient.reshape(self.input_shape)
