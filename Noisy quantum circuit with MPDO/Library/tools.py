"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensornetwork as tn
import torch as tc
import random
import string


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
                _edge.name = _edge.name.replace('qr', '')
            _axis_names.append(_edge.name)
        _node.axis_names = _axis_names

def ket0(dtype):
    r"""
    Return: Return the state |0>
    """
    return tc.tensor([1. + 0.j, 0. + 0.j], dtype=dtype)

def ket1(dtype):
    r"""
    Return: Return the state |1>
    """
    return tc.tensor([0. + 0.j, 1. + 0.j], dtype=dtype)

def ket_hadamard(dtype):
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=dtype)

def ket_plus(dtype):
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=dtype)

def ket_minus(dtype):
    r"""
    Return: Return the state |->
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), -1. / tc.sqrt(tc.tensor(2.))], dtype=dtype)

def create_ket0Series(number: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |0> * _number
    """

    _mps = [
        tn.Node(ket0(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket1Series(number: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |1> * _number
    """

    _mps = [
        tn.Node(ket1(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket_hadamardSeries(number: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_hadamard(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ketPlusSeries(number: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits;
        dtype

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_plus(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ketMinusSeries(number: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |-> * _number
    """

    _mps = [
        tn.Node(ket_minus(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ketRandomSeries(number: int, tensor: tc.Tensor, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits;
        tensor: the tensor to be used to create nodes;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |random> * _number
    """

    tensor = tensor.to(dtype=dtype)
    _mps = [
        tn.Node(tensor, name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
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
        _basis = tc.zeros((2 ** N, 1), dtype=tc.complex128)
        _basis[ii] = 1
        _view_basis.append(_basis)
    return _view_basis

def density2prob(rho_in: tc.Tensor, bases: list = None, basis_name: list = None, tolerant: float = 5e-4) -> dict:
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
    raise NotImplementedError('This function may cause error now.')
    _qn = int(np.log(rho_in.shape[0]) / np.log(2))

    if bases is None:
        bases = basis_list(_qn)

    _prob = []
    for ii in range(rho_in.shape[0]):
        _prob.append(tc_expect(rho_in, bases[ii]))

    # Form a dictionary
    if basis_name is None:
        basis_name = basis_name_list(_qn)
    _dc = {}
    for _i in range(len(basis_name)):
        _dc[basis_name[_i]] = tc.abs(_prob[_i])

    # Normalization
    _sum_result = 0
    for _value in _dc.values():
        _sum_result += _value
    for _name in _dc.keys():
        _dc[_name] = _dc[_name] / _sum_result

    # Remove prob. under threshold
    for _name in list(_dc.keys()):
        if _dc[_name] < tolerant:
            del _dc[_name]
            continue
    return _dc

def plot_histogram(prob_psi: dict, filename: str = None):
    r"""
    Plot a histogram of probability distribution.

    Args:
        prob_psi: probability of states, should be input as a dict;
        filename: location to save the fig, while None, it does not work.
    """
    if not isinstance(prob_psi, dict):
        raise TypeError('Prob distribution should be input as a dict, with keys as basis_name.')

    qnumber = len(list(prob_psi.keys())[0])

    plt.figure(figsize=(10, 8), dpi=300)
    plt.bar(prob_psi.keys(), prob_psi.values(), color='b')
    plt.ylim(ymax=1)
    plt.xticks(rotation=-45)
    plt.title(f'Probability distribution qnumber={qnumber}')
    plt.xlabel('State')
    plt.ylabel('Prob')

    if filename is not None:
        plt.savefig(filename)
    plt.show()

def select_device(device: str or int = 'cpu'):
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