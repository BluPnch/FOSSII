import json
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class RecurrentRegressor:
    def __init__(self, kind, input_size=5, hidden_size=32, seed=None):
        groups = {"RNN": 1, "GRU": 3, "LSTM": 4}
        if kind not in groups:
            raise ValueError(f"Unknown cell: {kind}")
        self.kind = kind
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)

        def initialize(shape, fan_in):
            bound = 1 / np.sqrt(fan_in)
            return rng.uniform(-bound, bound, shape).astype(np.float32)

        width = groups[kind] * hidden_size
        self.weights = {
            "W": initialize((input_size, width), input_size),
            "bx": initialize(width, input_size),
            "U": initialize((hidden_size, width), hidden_size),
            "bh": initialize(width, hidden_size),
            "Wy": initialize(hidden_size, hidden_size),
            "by": initialize(1, hidden_size),
        }

    def __call__(self, x):
        prediction, _, _ = self._forward(x)
        return prediction

    def _forward(self, x, keep_steps=False):
        p = self.weights
        
        h = np.zeros((len(x), self.hidden_size), dtype=x.dtype)
        c = np.zeros_like(h)
        steps = []

        # Линейный слой для x_t сразу на всех временных шагах.
        inputs = x @ p["W"] + p["bx"]
        for projected in inputs.swapaxes(0, 1):

            previous_h = h
            recurrent = previous_h @ p["U"] + p["bh"]

            if self.kind == "RNN":
                h = np.tanh(projected + recurrent)
                cache = (h,)

            elif self.kind == "GRU":
                xr, xz, xn = np.split(projected, 3, axis=-1)
                hr, hz, hn = np.split(recurrent, 3, axis=-1)
                
                reset = sigmoid(xr + hr)
                update = sigmoid(xz + hz)

                candidate = np.tanh(xn + reset * hn)
                
                saved = update * previous_h
                written = (1 - update) * candidate
                h = saved + written

                cache = reset, update, candidate, hn

            else:
                i, f, g, o = np.split(projected + recurrent, 4, axis=-1)
                i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
                previous_c = c

                written = i * g
                saved = f * previous_c
                c = saved + written

                memory = np.tanh(c)
                h = o * memory

                cache = i, f, g, o, previous_c, memory

            if keep_steps:
                steps.append((previous_h, cache))

        # Линейный выход из последнего h: один прогноз на каждое окно.
        return h @ p["Wy"] + p["by"], h, steps

    def loss_and_gradients(self, x, y):
        p = self.weights
        prediction, last_h, steps = self._forward(x, keep_steps=True)
        error = prediction - y

        dy = (2 / len(y)) * error
        gradients = {name: np.zeros_like(weight) for name, weight in p.items()}
        gradients["Wy"] = last_h.T @ dy
        gradients["by"][0] = dy.sum()
        dh = dy[:, None] * p["Wy"]
        dc = np.zeros_like(dh)

        # BPTT
        for t in range(len(steps) - 1, -1, -1):
            previous_h, cache = steps[t]
            if self.kind == "RNN":
                (h,) = cache

                dx_proj = dh_proj = dh * (1 - h**2)
                direct_dh = 0

            elif self.kind == "GRU":
                reset, update, candidate, hn = cache

                direct_dh = dh * update
                dz = dh * (previous_h - candidate) * update * (1 - update)
                dn = dh * (1 - update) * (1 - candidate**2)
                dr = dn * hn * reset * (1 - reset)
                dx_proj = np.concatenate((dr, dz, dn), axis=-1)
                dh_proj = np.concatenate((dr, dz, dn * reset), axis=-1)

            else:
                i, f, g, o, previous_c, memory = cache

                dc = dc + dh * o * (1 - memory**2)
                di = dc * g * i * (1 - i)
                df = dc * previous_c * f * (1 - f)
                dg = dc * i * (1 - g**2)
                do = dh * memory * o * (1 - o)
                dc = dc * f
                dx_proj = dh_proj = np.concatenate((di, df, dg, do), axis=-1)
                direct_dh = 0

            gradients["W"] += x[:, t, :].T @ dx_proj
            gradients["bx"] += dx_proj.sum(axis=0)
            gradients["U"] += previous_h.T @ dh_proj
            gradients["bh"] += dh_proj.sum(axis=0)
            dh = direct_dh + dh_proj @ p["U"].T

        return float(np.mean(error**2)), gradients

    def save(self, path, preprocessing):
        metadata = dict(kind=self.kind, input_size=self.input_size, hidden_size=self.hidden_size, preprocessing=preprocessing)
        np.savez(path, metadata=json.dumps(metadata), **self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(checkpoint["metadata"].item())
            model = cls(metadata["kind"], metadata["input_size"], metadata["hidden_size"])
            for name, weight in model.weights.items():
                weight[...] = checkpoint[name]
        return model, metadata


class Adam:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.learning_rate = learning_rate
        self.first = {name: np.zeros_like(value) for name, value in weights.items()}
        self.second = {name: np.zeros_like(value) for name, value in weights.items()}
        self.step_count = 0

    def step(self, gradients):
        # Ограничиваем общую норму градиента единицей.
        norm = np.sqrt(sum(np.sum(grad**2, dtype=np.float64) for grad in gradients.values()))
        scale = min(1.0, 1 / max(norm, 1e-12))
        beta1, beta2 = 0.9, 0.999
        self.step_count += 1

        # Adam: накопление моментов, корректировка их смещения и обновление весов.
        for name, weight in self.weights.items():
            grad = gradients[name] * scale
            first, second = self.first[name], self.second[name]
            first *= beta1
            first += (1 - beta1) * grad
            second *= beta2
            second += (1 - beta2) * grad**2
            corrected_first = first / (1 - beta1**self.step_count)
            corrected_second = second / (1 - beta2**self.step_count)
            weight -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
