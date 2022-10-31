"""Optimization module."""
import math

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # setp
        for param in self.params:
            if param.grad is None:
                continue

            # NOTE: Apply weight decay firstly is very important to be consistent with PyTorc
            grad = param.grad.data + param.data * self.weight_decay

            u_t = self.u.get(param, 0)

            u_t_1 = self.momentum * u_t + (1 - self.momentum) * grad

            param.data -= self.lr * u_t_1
            self.u[param] = u_t_1  # may need to change key


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):

        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue

            # NOTE: Apply weight decay firstly is very important to be consistent with PyTorch
            grad = param.grad.data + param.data * self.weight_decay

            u_t = self.m.get(param, 0)
            v_t = self.v.get(param, 0)

            u_t_1 = self.beta1 * u_t + (1 - self.beta1) * grad
            v_t_1 = self.beta2 * v_t + (1 - self.beta2) * grad * grad

            u_t_1_hat = u_t_1 / (1 - math.pow(self.beta1, self.t))
            v_t_1_hat = v_t_1 / (1 - math.pow(self.beta2, self.t))

            grad_weight = u_t_1_hat / (ndl.power_scalar(v_t_1_hat, 0.5) + self.eps)

            param.data -= self.lr * grad_weight

            self.m[param] = u_t_1
            self.v[param] = v_t_1
