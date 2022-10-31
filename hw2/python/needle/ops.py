"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import List, Optional

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

from .autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return (out_grad[0] + out_grad[1],)


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return (out_grad,)


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
        return (a * self.scalar).astype(a.dtype)

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

    def compute(self, a: NDArray):
        return (a / self.scalar).astype(a.dtype)

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

        if len(input_shape) != len(self.shape):
            input_shape = tuple(
                [1] * (len(self.shape) - len(input_shape)) + list(input_shape)
            )

        axes = tuple(i for i, a in enumerate(input_shape) if a == 1)
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
            * Tensor(
                array_api.where(node.inputs[0].realize_cached_data() > 0, 1, 0).astype(
                    out_grad.dtype
                )
            ),
        )


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        if self.axes is None:
            max_z = array_api.amax(Z, self.axes)
            return (
                array_api.log(array_api.sum(array_api.exp(Z - max_z), self.axes))
                + max_z
            )

        if type(self.axes) not in (tuple, list):
            self.axes = (self.axes,)

        max_z = array_api.amax(Z, self.axes)
        orig_shape = Z.shape
        shape = [1 if i in self.axes else orig_shape[i] for i in range(len(orig_shape))]

        max_z_reshape = array_api.broadcast_to(max_z.reshape(shape), Z.shape)

        return (
            array_api.log(array_api.sum(array_api.exp(Z - max_z_reshape), self.axes))
            + max_z
        )

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        max_z = Tensor(array_api.amax(input.realize_cached_data(), self.axes))

        if self.axes is None:
            temp = exp(input - max_z)
            return (temp / summation(temp, self.axes) * out_grad,)

        if type(self.axes) not in (tuple, list):
            self.axes = (self.axes,)

        orig_shape = input.shape
        shape = [1 if i in self.axes else orig_shape[i] for i in range(len(orig_shape))]

        temp = exp(input - max_z.reshape(shape).broadcast_to(input.shape))

        return (
            temp
            / summation(temp, self.axes).reshape(shape).broadcast_to(input.shape)
            * out_grad.reshape(shape).broadcast_to(input.shape),
        )


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
