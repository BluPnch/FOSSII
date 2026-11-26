import numpy as np

from layers import KERNEL, PADDING, STRIDE, Conv2D, ConvTranspose2D, Linear, Rectifier, Reshape, Trainable, conv_output_size


LATENT_DIM = 32


class Network:
    def __init__(self, **layers):
        # В прямом проходе работают все слои. Оптимизатор получает параметры только обучаемых.
        self.layers = tuple(layers.values())
        self.trainable = {name: layer for name, layer in layers.items() if isinstance(layer, Trainable)}

    def forward(self, x, retain=False):
        # Входом следующего слоя становится выход предыдущего. retain сохраняет данные для backward.
        for layer in self.layers:
            x = layer.forward(x, retain)
        return x

    def backward(self, gradient, *, parameter_grad=True, input_grad=False):
        # Идём от последнего слоя к первому. Каждый слой передаёт градиент своему входу.
        # В самом первом слое рассчитываем градиент входа, только если его запросил вызывающий код.
        for index in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[index].backward(gradient, parameter_grad, input_grad or index > 0)
        return gradient

    def parameters(self):
        for layer in self.trainable.values():
            yield from layer.parameters()

    def weights(self):
        # Имена слоёв связывают массивы весов и смещений с записями в чекпойнте.
        return {
            f"{name}.{kind}": getattr(layer, kind)
            for name, layer in self.trainable.items()
            for kind in ("weight", "bias")
        }


class Encoder(Network):
    def __init__(
        self, rng, *, second_padding=PADDING, second_kernel_size=KERNEL, second_stride=STRIDE, latent_dim=LATENT_DIM,
    ):
        # После первой свёртки размер 14×14. Размер после второй зависит от её P, K и шага.
        map_size = conv_output_size(14, second_padding, second_kernel_size, second_stride)
        super().__init__(
            # Из единственного канала яркости 28×28 получаем 16 карт 14×14.
            conv1=Conv2D(1, 16, rng),
            relu1=Rectifier(),
            # Каждый из 32 новых фильтров читает все 16 карт. При окне 4×4 это 256 произведений
            # для одной выходной клетки. По умолчанию получаем 32 карты 7×7.
            conv2=Conv2D(16, 32, rng, padding=second_padding, kernel_size=second_kernel_size, stride=second_stride),
            relu2=Rectifier(),
            # Собираем карты в вектор h из 1 568 значений при размере 7×7. Каждый линейный выход читает весь h.
            # При latent_dim=32 выходы слоя задают 32 средних и 32 логарифма дисперсий.
            flatten=Reshape(-1),
            posterior=Linear(32 * map_size**2, 2 * latent_dim, rng),
        )


class Generator(Network):
    def __init__(self, rng, *, latent_dim=LATENT_DIM):
        super().__init__(
            # Из компактного кода z строим 1 568 значений и раскладываем их в 32 карты 7×7.
            projection=Linear(latent_dim, 32 * 7 * 7, rng),
            relu1=Rectifier(),
            reshape=Reshape(7, 7, 32),
            # Расширяем карты 7×7 → 14×14 → 28×28. Последняя карта содержит логиты яркости.
            upsample1=ConvTranspose2D(32, 16, rng),
            relu2=Rectifier(),
            upsample2=ConvTranspose2D(16, 1, rng),
        )


class Discriminator(Network):
    def __init__(self, rng):
        super().__init__(
            conv1=Conv2D(1, 16, rng),
            # Наклон 0,2 даёт градиент и на отрицательной ветви Leaky ReLU.
            relu1=Rectifier(0.2),
            conv2=Conv2D(16, 32, rng),
            relu2=Rectifier(0.2),
            flatten=Reshape(-1),
            # Из всех признаков получаем один логит подлинности для каждого изображения.
            classifier=Linear(32 * 7 * 7, 1, rng),
        )


class Adam:
    def __init__(self, parameters, learning_rate):
        self.parameters = tuple(parameters)
        self.learning_rate = learning_rate
        # Ручной Adam хранит для каждого параметра скользящее среднее градиентов и их квадратов.
        self.first = [np.zeros_like(weight) for weight, _ in self.parameters]
        self.second = [np.zeros_like(weight) for weight, _ in self.parameters]
        self.step_count = 0

    def step(self):
        beta1, beta2 = 0.9, 0.999
        self.step_count += 1
        for (weight, grad), first, second in zip(self.parameters, self.first, self.second):
            # Обновляем обе оценки с учётом градиента текущего батча.
            first *= beta1
            first += (1 - beta1) * grad
            second *= beta2
            second += (1 - beta2) * grad**2
            # Исправляем смещение оценок к нулю из-за нулевой инициализации.
            first_corrected = first / (1 - beta1**self.step_count)
            second_corrected = second / (1 - beta2**self.step_count)
            # Делим сглаженный градиент на оценку его масштаба. 1e-8 защищает деление от нуля.
            weight -= self.learning_rate * first_corrected / (np.sqrt(second_corrected) + 1e-8)
