"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import numpy as np

class TensorGate(object):
    def __init__(self):
        self.name = None
        self.tensor = None
        self.rank = None
        self.dimension = None
        self.single = None
        self.axis_name = None
        self.dtype = tc.complex128

    def cnot(self):
        self.name = 'CNOT'
        self.tensor = tc.zeros((2, 2, 2, 2), dtype=self.dtype)  # rank-4 tensor
        # Rank-4 tensor CNOT is constructed by its truth table
        self.tensor[0, 0, 0, 0] = 1  #
        self.tensor[0, 1, 0, 1] = 1
        self.tensor[1, 0, 1, 1] = 1
        self.tensor[1, 1, 1, 0] = 1
        self.rank = 4
        self.dimension = 2
        self.single = False
        self.axis_name = ['inner_0', 'inner_1', 'physics_0', 'physics_1']
        return self

    def x(self):
        self.name = 'X'
        self.tensor = tc.tensor([[0, 1], [1, 0]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def y(self):
        self.name = 'Y'
        self.tensor = tc.tensor([[0, -1j], [1j, 0]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def z(self):
        self.name = 'Z'
        self.tensor = tc.tensor([[1, 0], [0, -1]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def s(self):
        self.name = 'S'
        self.tensor = tc.tensor([[1, 0], [0, 1j]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def t(self):
        self.name = 'T'
        self.tensor = tc.tensor([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def h(self):
        self.name = 'H'
        self.tensor = tc.tensor([[1, 1], [1, -1]], dtype=self.dtype) / np.sqrt(2)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def rx(self, theta):
        self.name = 'RX'
        self.tensor = tc.tensor(
            [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                                 [-1j * np.sin(theta / 2), np.cos(theta / 2)]]
            , dtype=self.dtype
        )
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def ry(self, theta):
        self.name = 'RY'
        self.tensor = tc.tensor([[np.cos(theta / 2), -np.sin(theta / 2)],
                                 [np.sin(theta / 2), np.cos(theta / 2)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def rz(self, theta):
        self.name = 'RZ'
        self.tensor = tc.tensor([[np.exp(-1j * theta / 2), 0],
                                 [0, np.exp(1j * theta / 2)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def u1(self, theta):
        self.name = 'U1'
        self.tensor = tc.tensor([[1, 0], [0, np.exp(1j * theta)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def u2(self, phi, lam):
        self.name = 'U2'
        self.tensor = tc.tensor([[1, -np.exp(1j * lam)],
                                 [np.exp(1j * phi), np.exp(1j * (phi + lam))]]) / np.sqrt(2)
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def u3(self, theta, phi, lam):
        self.name = 'U3'
        self.tensor = tc.tensor([[np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                                 [np.exp(1j * phi) * np.sin(theta / 2),
                                  np.exp(1j * (phi + lam)) * np.cos(theta / 2)]])
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def u(self, theta, phi, lam, gamma):
        self.name = 'U'
        self.tensor = tc.tensor([[np.cos(theta / 2), -np.exp(1j * (lam + gamma)) * np.sin(theta / 2)],
                                 [np.exp(1j * (phi + gamma)) * np.sin(theta / 2),
                                  np.exp(1j * (phi + lam + gamma)) * np.cos(theta / 2)]])
        self.rank = 2
        self.dimension = 2
        self.single = True
        self.axis_name = ['inner', 'physics']
        return self

    def cz(self):
        self.name = 'CZ'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]], dtype=self.dtype).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = 2
        self.single = False
        self.axis_name = ['inner_0', 'inner_1', 'physics_0', 'physics_1']
        return self

    def swap(self):
        self.name = 'SWAP'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=self.dtype).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = 2
        self.single = False
        self.axis_name = ['inner_0', 'inner_1', 'physics_0', 'physics_1']
        return self

    def rzz(self, _theta):
        self.name = 'RZZ'
        self.tensor = tc.tensor(
            [
                [np.exp(-1j * _theta), 0, 0, 0],
                [0, np.exp(1j * _theta), 0, 0],
                [0, 0, np.exp(1j * _theta), 0],
                [0, 0, 0, np.exp(-1j * _theta)],
            ], dtype=self.dtype
        ).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = 2
        self.single = False
        self.axis_name = ['inner_0', 'inner_1', 'physics_0', 'physics_1']
        return self
