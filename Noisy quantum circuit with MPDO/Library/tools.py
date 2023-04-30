"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import itertools
import random
import string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensornetwork as tn
import torch as tc
from scipy.io import loadmat
from numpy.linalg import eig
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

def create_ket0Series(qnumber: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |0> * _number
    """

    _mps = [
        tn.Node(ket0(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket1Series(qnumber: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |1> * _number
    """

    _mps = [
        tn.Node(ket1(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket_hadamardSeries(qnumber: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_hadamard(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ketPlusSeries(qnumber: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_plus(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ketMinusSeries(qnumber: int, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |-> * _number
    """

    _mps = [
        tn.Node(ket_minus(dtype), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(qnumber)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ketRandomSeries(qnumber: int, tensor: tc.Tensor, dtype=tc.complex128) -> list:
    r"""
    create initial qubits

    Args:
        qnumber: the number of qubits;
        tensor: the tensor to be used to create nodes;
        dtype: the data type of the tensor.

    Returns:
        _mps: the initial mps with the state |random> * _number
    """

    tensor = tensor.to(dtype=dtype)
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

def readExpChi(filename: str = None):
    if filename is None:
        filename = './data/chi/chi1.mat'

    if '.mat' in filename:
        data = loadmat(filename)['exp']
        return data
    elif '.npz' in filename:
        data = np.load(filename)['chi']
        return data
    else:
        raise TypeError('Current file-type is not supported.')

def czNoisyTensor(chi, gate_factor: dict = None, dtype=tc.complex128, device: str or int = 'cpu') -> list[tc.Tensor]:
    r"""
    :param chi: Chi matrix from experiment;
    :param gate_factor: Manual API;
    :param dtype: Data type;
    :param device: Device;
    :return: qutip.Qobj

    Additional Information:
        Actual_space_size = int(np.log10(chi.shape[0]) / np.log10(4))
        N >= Actual_space_size:
            A bigger complete space can always find a subspace to describe smaller system.
        N < Actual_space_size:
            raise Error that space for describing such system is not complete.
    """
    def _czNoisySPara(_chi):
        r"""
        Get $ E_i = \sqrt{d_i}\sum_jU_{ji}\tilde{E}_j $
                where \tilde{E}_j represents gate basis
        :param _chi: Chi Matrix;
        :return: Unitary operations with linear combination coefficients.

        Sups:
            E --> {E_0: {-/}, E_1: {-/}, ...}
                    ==> E_0 --> {'I': Coefficient, ...},
                        E_0 --> {'I': Coefficient, ...},
                        ...
        """

        def __sortChiEig(_chi) -> tuple:
            r"""
            From a Chi matrix get its decomposition,
                                which is $ \chi = \sum_{xy}U_{mx}d_x\delta_xU^\dagger_{yn} $
                                         $ E_i = \sqrt{d_i}\sum_jU_{ji}\tilde{E}_j $.
            :return: chi's corresponding eigenvector and eigenvalue.
            """
            # Get eigenvalues and eigenvectors with descending order
            _s, _u = eig(_chi)
            _idx_ = _s.argsort()[::-1]
            _s = _s[_idx_]
            _u = _u[:, _idx_]
            return _u, _s

        def __opEIdx(_chi, _svd_error_tolerant: float = 1e-12) -> tuple:
            r"""

            :param _chi: Chi Matrix from experiment;
            :param _svd_error_tolerant: math in matrix operation may cause little fault or inaccuracy like 1e-33;
            :return: e as coefficient matrix(vector) for $\tilde{E}_j$;
                            non_zero_indices as idx of non-zero coefficient for zipping with its basis name in next step.
            """
            _u, _s = __sortChiEig(_chi)
            _effect_s = []
            for __value in _s:
                if np.abs(__value) > _svd_error_tolerant:
                    _effect_s.append(__value)
            # Get \vec{e}
            _e_ = [[]] * len(_effect_s)
            for _i_ in range(len(_effect_s)):
                _e_[_i_] = list(_u.T[_i_])
                for _ii_ in range(len(_e_[_i_])):
                    _e_[_i_][_ii_] = _e_[_i_][_ii_] * np.sqrt(_s[_i_])
                    if np.abs(_e_[_i_][_ii_]) < _svd_error_tolerant:
                        _e_[_i_][_ii_] = 0 + 0j
            #
            if len(_e_) == 1:  # Indicates only one eigenvalue exists
                non_zero_indices = []
                for _i_ in range(len(_e_[0])):
                    if _e_[0][_i_] != 0:
                        non_zero_indices.append(_i_)
            else:  # Not tested yet
                try:
                    non_zero_indices = [[]] * len(_e_)
                    for _i_ in range(len(_e_)):
                        for _ii_ in range(len(_e_[_i_])):
                            if _e_[_i_][_ii_] != 0:
                                non_zero_indices[_i_].append(_ii_)
                except:
                    raise IndexError('IndexError may take appear, codes here is not check yet.')
            return _e_, non_zero_indices

        _N = int(np.log10(_chi.shape[0]) / np.log10(4))

        _g_set = gates_list(_N)
        _e_matrix, index_list = __opEIdx(_chi)
        _effGate = {}
        if len(_e_matrix) == 1:
            for _idx in index_list:
                _effGate[_g_set[_idx]] = _e_matrix[0][_idx]
        else:
            for _i in range(len(_e_matrix)):
                _effGate['E_{}'.format(_i)] = {}
                for _idx in index_list[_i]:
                    _effGate['E_{}'.format(_i)][_g_set[_idx]] = _e_matrix[_i][_idx]
        return _effGate

    def is_nested_dict(d: dict):
        """
        check whether a dict is nested, True: { { } } -- False: { }.
        """
        return any(isinstance(j, dict) for j in d.values())

    if gate_factor is None:
        gate_factor = _czNoisySPara(chi)

    if is_nested_dict(gate_factor) is True:     # Multiple Error generator E_{} exist
        _ii, _tensor = 0, [tc.zeros((4, 4), dtype=dtype, device=device)] * len(gate_factor)
        for _value in gate_factor.values():
            for _name_, _value_ in _value.items():
                # linear equations
                _tensor[_ii] += tc.tensor(_value_, dtype=dtype, device=device) * name2matrix(_name_)
            _ii += 1
    else:
        _tensor = tc.tensor(0)
        for _name_, _value_ in gate_factor.items():
            # linear equations
            _tensor += tc.tensor(_value_, dtype=dtype, device=device) * name2matrix(_name_)
        _tensor = [_tensor]

    return _tensor

def name2matrix(operation_name: str, dtype=tc.complex128, device: str or int = 'cpu'):
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