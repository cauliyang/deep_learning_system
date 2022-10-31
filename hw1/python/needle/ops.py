"""Operator implementations."""

from itertools import zip_longest
from numbers import Number
from typing import List, Optional

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy
import numpy as array_api

from .autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * (self.scalar * node.inputs[0] ** (self.scalar - 1)),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axes = (-2, -1) if self.axes is None else self.axes
        return array_api.swapaxes(a, *axes)

    def gradient(self, out_grad, node):
        axes = (-2, -1) if self.axes is None else self.axes
        return (out_grad.transpose(axes),)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return (out_grad.reshape(node.inputs[0].shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = tuple(
            i
            for i, (a, b) in enumerate(zip_longest(input_shape, self.shape))
            if a == 1 or a is None
        )
        return (out_grad.sum(axes).reshape(input_shape),)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        if self.axes is None:
            return (out_grad.broadcast_to(node.inputs[0].shape),)

        if type(self.axes) not in (tuple, list):
            self.axes = (self.axes,)
        orig_shape = node.inputs[0].shape
        shape = [1 if i in self.axes else orig_shape[i] for i in range(len(orig_shape))]
        return (out_grad.reshape(shape).broadcast_to(node.inputs[0].shape),)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        if len(lhs.shape) > len(rhs.shape):
            axes = tuple(i for i in range(len(lhs.shape) - len(rhs.shape)))
            return out_grad.matmul(rhs.transpose()), lhs.transpose().matmul(
                out_grad
            ).sum(axes)
        elif len(lhs.shape) < len(rhs.shape):

            axes = tuple(i for i in range(len(rhs.shape) - len(lhs.shape)))

            return out_grad.matmul(rhs.transpose()).sum(axes), lhs.transpose().matmul(
                out_grad
            )

        return out_grad.matmul(rhs.transpose()), lhs.transpose().matmul(out_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return (-out_grad,)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return (out_grad / node.inputs[0],)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return (out_grad * exp(node.inputs[0]),)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return (
            out_grad
            * Tensor(array_api.where(node.inputs[0].realize_cached_data() > 0, 1, 0)),
        )


def relu(a):
    return ReLU()(a)
