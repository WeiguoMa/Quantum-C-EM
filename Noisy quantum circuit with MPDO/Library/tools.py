"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import collections
import itertools
import random
import string
import warnings
from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensornetwork as tn
import torch as tc
from scipy.optimize import minimize

from Library.TensorOperations import tensorDot


def is_nested(_lst: list) -> bool:
    r"""
    Check if a list is nested

    Args:
        _lst: the list to be checked

    Returns:
        True if the list is nested, False otherwise
    """
    return any(isinstance(_i, list) for _i in _lst)


def EdgeName2AxisName(_nodes: list[tn.Node] or list[tn.AbstractNode]):
    r"""
    ProcessFunction -->
        In tensornetwork package, axis_name is not equal to _name_of_edge_. While calculating, to ensure that
                we are using right order of axis_names, we need to set axis_names of _gate according to its edges' name.

    Args:
        _nodes: the node to be set axis_names.

    Returns:
        None, but the axis_names of _nodes will be set in memory.
    """
    if not isinstance(_nodes, list):
        if not isinstance(_nodes, tn.Node) and not isinstance(_nodes, tn.AbstractNode):
            raise ValueError('The input should be a list of nodes.')
        _nodes = [_nodes]

    for _node in _nodes:
        _axis_names = []
        for _edge in [_node[i] for i in range(_node.get_rank())]:
            # hardcode, which is relating to code design from weiguo
            if 'qr' in _edge.name:
                _edge.set_name(_edge.name.replace('qr', ''))
            if 'bond_' in _edge.name:  # Fact that 'bond_a_b' is the same as 'bond_b_a'
                _split = _edge.name.split('_')
                if int(_split[1]) > int(_split[2]):
                    _edge.name = f'bond_{_split[2]}_{_split[1]}'
            _axis_names.append(_edge.name)
        _node.axis_names = _axis_names


def ket0(dtype, device: str or int = 'cpu'):
    r"""
    Return: Return the state |0>
    """
    return tc.tensor([1. + 0.j, 0. + 0.j], dtype=dtype, device=device)


def ket1(dtype, device: str or int = 'cpu'):
    r"""
    Return: Return the state |1>
    """
    return tc.tensor([0. + 0.j, 1. + 0.j], dtype=dtype, device=device)


def ket_hadamard(dtype, device: str or int = 'cpu'):
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=dtype, device=device)


def ket_plus(dtype, device: str or int = 'cpu'):
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=dtype, device=device)


def ket_minus(dtype, device: str or int = 'cpu'):
    r"""
    Return: Return the state |->
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), -1. / tc.sqrt(tc.tensor(2.))], dtype=dtype, device=device)


def create_ket0Series(qnumber: int, dtype=tc.complex64, device: str or int = 'cpu') -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor;
        device: cpu or gpu.

    Returns:
        _mps: the initial mps with the state |0> * _number
    """

    _mps = [
        tn.Node(ket0(dtype, device=device), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps


def create_ket1Series(qnumber: int, dtype=tc.complex64, device: str or int = 'cpu') -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor;
        device: cpu or gpu.

    Returns:
        _mps: the initial mps with the state |1> * _number
    """

    _mps = [
        tn.Node(ket1(dtype, device=device), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps


def create_ketHadamardSeries(qnumber: int, dtype=tc.complex64, device: str or int = 'cpu') -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor;
        device: cpu or gpu.

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_hadamard(dtype, device=device), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps


def create_ketPlusSeries(qnumber: int, dtype=tc.complex64, device: str or int = 'cpu') -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor;
        device: cpu or gpu.

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_plus(dtype, device=device), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps


def create_ketMinusSeries(qnumber: int, dtype=tc.complex64, device: str or int = 'cpu') -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor;
        device: cpu or gpu.

    Returns:
        _mps: the initial mps with the state |-> * _number
    """

    _mps = [
        tn.Node(ket_minus(dtype, device=device), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps


def create_ketRandomSeries(qnumber: int, tensor: tc.Tensor, dtype=tc.complex64, device: str or int = 'cpu') -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        tensor: the tensor to be used to create nodes;
        dtype: the data type of the tensor;
        device: cpu or gpu.

    Returns:
        _mps: the initial mps with the state |random> * _number
    """

    tensor = tensor.to(dtype=dtype, device=device)
    _mps = [
        tn.Node(tensor, name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps


def plot_nodes(_nodes):
    r"""
    Plot tensor network nodes.

    Args:
        _nodes: nodes to be plotted.

    Returns:
        None
    """
    raise NotImplementedError('Plotting is not supported yet.')


def tc_expect(operator: tc.Tensor, state: tc.Tensor) -> tc.Tensor:
    if not isinstance(operator, tc.Tensor) or not isinstance(state, tc.Tensor):
        raise TypeError('torch.Tensor should be input')
    if state.shape[0] == state.shape[1]:
        matrix = tc.matmul(state, operator)
        return tc.abs(tc.sum(tc.diag(matrix)))
    else:
        if state.shape[0] == 1:
            # state is row
            state.reshape((state.shape[0], 1))
        result = tc.matmul(state.T.conj(), tc.matmul(operator, state))
        return tc.abs(result)


def basis_name_list(N: int) -> list:
    r"""
    Generate a series of bases' name, like
            N = 2, ['00', '01', '10', '11']
    """
    _binary_rep = ['0', '1']
    _b_set = [''.join(ii) for ii in itertools.product(_binary_rep, repeat=N)]
    return _b_set


def basis_list(N: int) -> list:
    r"""
    Generate a series of bases, like
                |00> = tensor([basis(2, 0), basis(2, 0)])
                |10> = tensor([basis(2, 1), basis(2, 0)])

    Attention:
            different from qutip.basis(4, 0), which != |00>

    Notice:
        Author didn't find whether there exists another fast way to generate such basis set,
            main goal is to get probability distribution from a density matrix.
                That is, p_{basis} = qutip.expect(density_matrix, basis)
                            --> p = <\psi|density_matrix|\psi>
    """
    _view_basis = []
    for ii in range(2 ** N):
        _basis = tc.zeros((2 ** N, 1), dtype=tc.complex64)
        _basis[ii] = 1
        _view_basis.append(_basis)
    return _view_basis


def density2prob(rho_in: tc.Tensor, bases: list = None, basis_name: list = None,
                 tolerant: Optional[float] = None) -> dict:
    r"""
    Transform density matrix into probability distribution with provided bases.

    Args:
        rho_in: density matrix;
        bases: provided projected bases;
        basis_name: name of bases, like '00000';
        tolerant: probability under this threshold will not be shown.

    Additional information:
        function utilized:
                tc_expect()
    """
    _qn = int(np.log(rho_in.shape[0]) / np.log(2))

    if bases is None:
        bases = basis_list(_qn)

    _prob = []
    for _ii in range(len(bases)):
        _prob.append(float(tc_expect(rho_in, bases[_ii])))

    # Form a dictionary
    if basis_name is None:
        basis_name = basis_name_list(_qn)

    _dc = {}
    for _i in range(len(basis_name)):
        _dc[basis_name[_i]] = _prob[_i]

    # Normalization
    _sum_result = 0
    for _value in _dc.values():
        _sum_result += _value
    for _name in _dc.keys():
        _dc[_name] = _dc[_name] / _sum_result

    if tolerant is not None:
        # Remove prob. under threshold
        for _name in list(_dc.keys()):
            if _dc[_name] < tolerant:
                del _dc[_name]
                continue
    return _dc


def plot_histogram(prob_psi: dict, title: str = None, filename: str = None):
    r"""
    Plot a histogram of probability distribution.

    Args:
        prob_psi: probability of states, should be input as a dict;
        title: title of the fig, while None, it does not work;
        filename: location to save the fig, while None, it does not work.
    """
    if not isinstance(prob_psi, dict):
        raise TypeError('Prob distribution should be input as a dict, with keys as basis_name.')

    qnumber = len(list(prob_psi.keys())[0])

    plt.figure(figsize=(10, 8), dpi=300)
    plt.bar(prob_psi.keys(), prob_psi.values(), color='b')
    plt.ylim(ymax=1)
    plt.xticks(rotation=-45)

    if title is None:
        plt.title(f'Probability distribution qnumber={qnumber}')
    else:
        plt.title(title)

    plt.xlabel('State')
    plt.ylabel('Prob')

    if filename is not None:
        plt.savefig(filename)
    plt.show()


def select_device(device: str or int = None):
    if isinstance(device, str):
        return device
    else:
        if tc.cuda.is_available():
            if device is None:
                return 'cuda:0'
            else:
                return f'cuda:{device}'
        else:
            warnings.warn('CUDA is not available, use CPU instead.')
            return 'cpu'


def generate_random_string_without_duplicate(_n: int):
    r"""
    Generate a random string without duplicate characters.

    Args:
        _n: The length of the string.

    Returns:
        _str: The random string.
    """

    def _generate_random_string(_n_):
        return ''.join(random.choices(string.ascii_lowercase, k=_n_))

    _str = _generate_random_string(_n)
    while len(_str) != len(set(_str)):
        _str = _generate_random_string(_n)
    return _str


def move_index(_str: str, _idx1: int, _idx2: int):
    r"""
    Move the index from _idx1 to _idx2.

    Args:
        _str: The string;
        _idx1: The index to be moved;
        _idx2: The index to be moved to.

    Returns:
        _str: The string after moving the index.
    """
    _str = list(_str)
    _str.insert(_idx2, _str.pop(_idx1))
    return ''.join(_str)


def gates_list(N: int, basis_gates: list = None) -> list:
    r"""
    Generates a series of gate sets as basis,
    N = 1 --> list[I, X, Y, Z]
    N = 2 --> list[II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
    ... Strings are inside of list.
    Attention:
        This function ONLY fit well A CAPITAL as a gate. Ugly but useful
                                        , u might found that this is just a representation.
    :param N: Number of qubits;
    :param basis_gates: basis you chose as basis;
    :return: Basis list.
    """
    if basis_gates is None:
        basis_gates = ['I', 'X', 'Y', 'Z']
    g_set = [''.join(i) for i in itertools.product(basis_gates, repeat=N)]
    return g_set


def name2matrix(operation_name: str, dtype=tc.complex64, device: str or int = 'cpu'):
    r"""
    As you can see below, A CAPITAL stands for a basis, actually an operation, that is arbitrarily defined.
    :param operation_name: like 'ZZZ'
    :param dtype: data type
    :param device: cpu or gpu
    :return: Product matrix
    """
    operation_list = []
    for letter in operation_name:
        if letter == 'I':
            operation_list.append(tc.eye(2, dtype=dtype, device=device))
        elif letter == 'X':
            operation_list.append(tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device))
        elif letter == 'Y':
            operation_list.append(-1j * tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device))
        elif letter == 'Z':
            operation_list.append(tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device))
    return tensorDot(operation_list)


def find_duplicate(_lst_):
    """ Find the duplicate items and their indices in a list """
    _duplicate_item_ = [item for item, count in collections.Counter(_lst_).items() if count > 1]
    _duplicate_idx_ = [idx for idx, item in enumerate(_lst_) if item in _duplicate_item_]
    return _duplicate_item_, _duplicate_idx_


def sqrt_matrix(density_matrix):
    r"""Compute the square root matrix of a density matrix where :math:`\rho = \sqrt{\rho} \times \sqrt{\rho}`

    Args:
        density_matrix (tensor_like): 2D density matrix of the quantum system.

    Returns:
        (tensor_like): Square root of the density matrix.
    """
    evs, vecs = tc.linalg.eigh(density_matrix)
    evs = tc.where(evs > 0.0, evs, 0.0)
    evs = tc.real(evs).to(tc.complex64)
    return vecs @ tc.diag(tc.sqrt(evs)) @ vecs.T.conj()


def cal_fidelity(rho: tc.Tensor, sigma: tc.Tensor) -> tc.Tensor:
    """
    Calculate the fidelity between two density matrices.

    Equation:
        F(rho, sigma) = Tr(\sqrt{\sqrt{rho} sigma \sqrt{rho}})

    Args:
        rho: density matrix;
        sigma: density matrix.
    """
    if rho.shape != sigma.shape:
        raise ValueError('The shape of rho and sigma should be equal.')
    if rho.shape[0] != rho.shape[1] or rho.shape[0] != sigma.shape[1]:
        raise ValueError('The shape of rho and sigma should be square.')

    _sqrt_rho = sqrt_matrix(rho)
    _sqrt_rho_sigma_sqrt_rho = _sqrt_rho @ sigma @ _sqrt_rho

    evs = tc.linalg.eigvalsh(_sqrt_rho_sigma_sqrt_rho)
    evs = tc.real(evs)
    evs = tc.where(evs > 0.0, evs, 0.0)

    trace = tc.sum(tc.sqrt(evs), -1)

    return trace


def validDensityMatrix(rho, methodIdx=1, constraints='eq', hermitian=True, device: str or int = 'cpu'):
    """
    Produced by Dr.Shi  --- Data Science

    Args:
        rho: density matrix;
        methodIdx: 0, 1, 2, refers to different scipy.optimal.minimize methods;
        constraints: 'eq' or 'ineq';
        hermitian: True or False;
        device: cpu or gpu.

    Returns:
        rho_semi: valid density matrix.

    """
    # rho = rho/np.trace(rho)
    if hermitian:
        rho = 0.5 * (rho + rho.T.conj())
    rho = rho.to(device='cpu')
    ps, psi = np.linalg.eigh(rho)

    traceV = 1.0  # tc.trace(rho)
    fitFunc = lambda p: np.sum(np.abs(p - ps) ** 2)
    bounds = [(0.0, traceV + 0.001) for idx in range(len(ps))]

    x0 = deepcopy(ps)
    x0[x0 < 0.0] = 0.0
    x0 = x0 / np.sum(x0) * traceV

    if constraints == 'eq':
        cons = ({'type': 'eq', 'fun': lambda p: np.sum(p) - traceV})
    elif constraints == 'ineq':
        cons = ({'type': 'ineq', 'fun': lambda p: traceV - np.sum(p)})
    else:
        raise ValueError('constraints should be eq or ineq')

    optMethods = ['L-BFGS-B', 'SLSQP', 'COBYLA']

    res = minimize(fitFunc, x0, method=optMethods[methodIdx], constraints=cons, bounds=bounds)
    newPs = res.x

    psi, mewPs = tc.tensor(psi), tc.tensor(newPs)

    rho_semi = psi @ np.diag(newPs) @ psi.T.conj()
    return rho_semi.to(device=device)
