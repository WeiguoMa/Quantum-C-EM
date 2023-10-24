"""
Author: weiguo_ma
Time: 04.25.2023
Contact: weiguo.m@iphy.ac.cn
"""
import numpy as np
import torch as tc

from Library.tools import select_device


class TensorGate(object):
    def __init__(self, ideal: bool = None, device: str or int = 0, dtype=tc.complex128):
        self.name = None
        self.tensor = None
        self.rank = None
        self.dimension = None
        self.single = None
        self.ideal = ideal

        self.device = select_device(device)
        self.dtype = dtype

    def i(self):
        self.name = 'I'
        self.tensor = tc.tensor([[1, 0], [0, 1]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def x(self):
        self.name = 'X'
        self.tensor = tc.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def y(self):
        self.name = 'Y'
        self.tensor = tc.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def z(self):
        self.name = 'Z'
        self.tensor = tc.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def s(self):
        self.name = 'S'
        self.tensor = tc.tensor([[1, 0], [0, 1j]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def t(self):
        self.name = 'T'
        self.tensor = tc.tensor([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def h(self):
        self.name = 'H'
        self.tensor = tc.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) / np.sqrt(2)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def ii(self):
        self.name = 'II'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    def cx(self):
        self.name = 'CX'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    def cy(self):
        self.name = 'CY'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, -1j],
                                 [0, 0, 1j, 0]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    def cz(self):
        self.name = 'CZ'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    def swap(self):
        self.name = 'SWAP'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    def cnot(self):
        self.name = 'CNOT'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    # Variational gates

    def rx(self, theta: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RX'
        self.tensor = tc.tensor(
            [[tc.cos(theta / 2), -1j * tc.sin(theta / 2)],
             [-1j * tc.sin(theta / 2), tc.cos(theta / 2)]]
            , dtype=self.dtype, device=self.device
        )
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def ry(self, theta: tc.Tensor or float = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RY'
        self.tensor = tc.tensor([[tc.cos(theta / 2), -tc.sin(theta / 2)],
                                 [tc.sin(theta / 2), tc.cos(theta / 2)]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def rz(self, theta: tc.Tensor or float = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RZ'
        self.tensor = tc.tensor([[tc.exp(-1j * theta / 2), 0],
                                 [0, tc.exp(1j * theta / 2)]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def u1(self, theta: tc.Tensor or float = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'U1'
        self.tensor = tc.tensor([[1, 0], [0, tc.exp(1j * theta)]], dtype=self.dtype, device=self.device)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def u2(self, phi: tc.Tensor or float = None, lam: tc.Tensor or float = None):
        if phi is None:
            phi = tc.randn(1, dtype=self.dtype)
        if lam is None:
            lam = tc.randn(1, dtype=self.dtype)
        if isinstance(phi, float) or isinstance(phi, int):
            phi = tc.tensor(phi, dtype=self.dtype)
        if isinstance(lam, float) or isinstance(lam, int):
            lam = tc.tensor(lam, dtype=self.dtype)
        phi = phi.to(device=self.device, dtype=self.dtype)
        lam = lam.to(device=self.device, dtype=self.dtype)

        self.name = 'U2'
        self.tensor = tc.tensor([[1, -tc.exp(1j * lam)],
                                 [tc.exp(1j * phi), tc.exp(1j * (phi + lam))]],
                                device=self.device, dtype=self.dtype) / np.sqrt(2)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def u3(self, theta: tc.Tensor or float = None, phi: tc.Tensor or float = None, lam: tc.Tensor or float = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if phi is None:
            phi = tc.randn(1, dtype=self.dtype)
        if lam is None:
            lam = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        if isinstance(phi, float) or isinstance(phi, int):
            phi = tc.tensor(phi, dtype=self.dtype)
        if isinstance(lam, float) or isinstance(lam, int):
            lam = tc.tensor(lam, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)
        phi = phi.to(device=self.device, dtype=self.dtype)
        lam = lam.to(device=self.device, dtype=self.dtype)

        self.name = 'U3'
        self.tensor = tc.tensor([[tc.cos(theta / 2), -tc.exp(1j * lam) * tc.sin(theta / 2)],
                                 [tc.exp(1j * phi) * tc.sin(theta / 2),
                                  tc.exp(1j * (phi + lam)) * tc.cos(theta / 2)]], device=self.device, dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def rzz(self, theta: tc.Tensor or float = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RZZ'
        self.tensor = tc.tensor(
            [
                [tc.exp(-1j * theta), 0, 0, 0],
                [0, tc.exp(1j * theta), 0, 0],
                [0, 0, tc.exp(1j * theta), 0],
                [0, 0, 0, tc.exp(-1j * theta)],
            ], dtype=self.dtype, device=self.device
        ).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self

    def arbGateSingle(self, tensor: tc.Tensor = None):
        """ Arbitrary Single qubit gate """
        if tensor is None:
            tensor = tc.randn((2, 2), dtype=self.dtype)
        if isinstance(tensor, np.ndarray):
            self.tensor = tc.tensor(tensor, dtype=self.dtype)
        elif isinstance(tensor, tc.Tensor):
            self.tensor = tensor.to(device=self.device, dtype=self.dtype)
        else:
            raise TypeError('Tensor must be a numpy array or a torch tensor')
        if tensor.shape != (2, 2):
            raise ValueError('Tensor must be of shape (2, 2)')

        self.name = 'ArbGateSingle'
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        return self

    def arbGateDouble(self, tensor: tc.Tensor = None):
        """ Arbitrary Double qubit gate """
        if tensor is None:
            tensor = tc.randn((2, 2, 2, 2), dtype=self.dtype)
        if isinstance(tensor, np.ndarray):
            self.tensor = tc.tensor(tensor, dtype=self.dtype)
        elif isinstance(tensor, tc.Tensor):
            self.tensor = tensor.to(device=self.device, dtype=self.dtype)
        else:
            raise TypeError('Tensor must be a numpy array or a torch tensor')
        if tensor.shape != (2, 2, 2, 2):
            raise ValueError('Tensor must be of shape (2, 2, 2, 2)')

        self.name = 'ArbGateDouble'
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        return self