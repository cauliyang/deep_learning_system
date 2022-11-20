"""The module."""
from typing import List

import needle.init as init
from needle import ops
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype).reshape(
                    shape=(1, self.out_features)
                )
            )

    def forward(self, X: Tensor) -> Tensor:
        if not self.bias:
            return X @ self.weight
        else:
            return X @ self.weight + self.bias.broadcast_to(
                shape=(*X.shape[:-1], self.out_features)
            )


class Flatten(Module):
    def forward(self, X):
        dims = 1
        for s in X.shape[1:]:
            dims *= s
        return X.reshape((X.shape[0], dims))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def append(self, module: Module):
        self.modules.append(module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)

        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        classes = logits.shape[-1]
        y_one_hot = init.one_hot(classes, y, device=logits.device, dtype=logits.dtype)
        return (
            ops.summation(
                ops.logsumexp(logits, axes=1) - ops.summation(logits * y_one_hot, axes=1)
            )
            / logits.shape[0]
        )


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))

        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:

        num_sample = x.shape[0]

        if self.training:
            expect_origin = ops.summation(x, 0) / num_sample
            expect = expect_origin.reshape((1, x.shape[1]))
            variance_origin = (
                ops.summation(ops.power_scalar(x - expect.broadcast_to(x.shape), 2), 0)
                / num_sample
            )
            variance = variance_origin.reshape((1, x.shape[1]))

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * expect_origin.data  # may need to use .data

            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * variance_origin.data  # may need to use .data

            return self.weight.broadcast_to(x.shape) * (
                (x - expect.broadcast_to(x.shape))
                / ops.power_scalar(variance.broadcast_to(x.shape) + self.eps, 0.5)
            ) + self.bias.broadcast_to(x.shape)

        else:
            expect = self.running_mean
            variance = self.running_var

            return (x - expect.broadcast_to(x.shape)) / ops.power_scalar(
                variance.broadcast_to(x.shape) + self.eps, 0.5
            ) * self.weight.data.broadcast_to(x.shape) + self.bias.data.broadcast_to(x.shape)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        expect = (ops.summation(x, 1) / self.dim).reshape((x.shape[0], 1))
        variance = (
            ops.summation(ops.power_scalar(x - expect.broadcast_to(x.shape), 2), 1) / self.dim
        ).reshape((x.shape[0], 1))

        return self.weight.broadcast_to(x.shape) * (
            (x - expect.broadcast_to(x.shape))
            / ops.power_scalar(variance.broadcast_to(x.shape) + self.eps, 0.5)
        ) + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)
            return x * mask / (1 - self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
