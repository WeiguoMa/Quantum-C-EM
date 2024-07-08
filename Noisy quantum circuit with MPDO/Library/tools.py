"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import itertools
import random
import string
import warnings
from copy import deepcopy
from functools import reduce
from typing import Optional, List, Union, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensornetwork as tn
import torch as tc
from scipy.optimize import minimize

mpl.rcParams['font.family'] = 'Arial'


def EdgeName2AxisName(_nodes: List[tn.AbstractNode]):
    r"""
    ProcessFunction -->
        In tensornetwork package, axis_name is not equal to _name_of_edge_. While calculating, to ensure that
                we are using right order of axis_names, we need to set axis_names of _gate according to its edges' name.

    Args:
        _nodes: the node to be set axis_names.

    Returns:
        None, but the axis_names of _nodes will be set in memory.
    """
    if not isinstance(_nodes, List):
        if not isinstance(_nodes, tn.AbstractNode):
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


def ket0(dtype, device: Union[str, int] = 'cpu'):
    r"""
    Return: Return the state |0>
    """
    return tc.tensor([1. + 0.j, 0. + 0.j], dtype=dtype, device=device)


def ket1(dtype, device: Union[str, int] = 'cpu'):
    r"""
    Return: Return the state |1>
    """
    return tc.tensor([0. + 0.j, 1. + 0.j], dtype=dtype, device=device)


def ket_hadamard(dtype, device: Union[str, int] = 'cpu'):
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=dtype, device=device)


def ket_plus(dtype, device: Union[str, int] = 'cpu'):
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=dtype, device=device)


def ket_minus(dtype, device: Union[str, int] = 'cpu'):
    r"""
    Return: Return the state |->
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), -1. / tc.sqrt(tc.tensor(2.))], dtype=dtype, device=device)


def create_ket0Series(qnumber: int, dtype=tc.complex64, device: Union[str, int] = 'cpu') -> list:
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


def create_ket1Series(qnumber: int, dtype=tc.complex64, device: Union[str, int] = 'cpu') -> list:
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


def create_ketHadamardSeries(qnumber: int, dtype=tc.complex64, device: Union[str, int] = 'cpu') -> list:
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


def create_ketPlusSeries(qnumber: int, dtype=tc.complex64, device: Union[str, int] = 'cpu') -> list:
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


def create_ketMinusSeries(qnumber: int, dtype=tc.complex64, device: Union[str, int] = 'cpu') -> list:
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


def create_ketRandomSeries(qnumber: int, tensor: tc.Tensor, dtype=tc.complex64,
                           device: Union[str, int] = 'cpu') -> list:
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


def tc_expect(oper: tc.Tensor, state: tc.Tensor) -> tc.Tensor:
    """
        Calculates the expectation value for operator(s) and state(s) in PyTorch,
        using torch.matmul for matrix multiplication.

        Parameters
        ----------
        oper : torch.Tensor/list
            A single or a `list` of operators for expectation value.

        state : torch.Tensor/list
            A single or a `list` of quantum state vectors or density matrices.

        Returns
        -------
        expt : torch.Tensor
            Expectation value as a PyTorch tensor. Complex if `oper` is not Hermitian.
        """

    def _single_expect(o, s):
        if s.dim() == 1:  # State vector (ket)
            return tc.einsum('i, ij, j', s.conj(), o, s)
        elif s.dim() == 2:  # Density matrix
            return tc.trace(tc.matmul(o, s))
        else:
            raise ValueError("State must be a vector or matrix.")

    # Handling a single operator and state
    if isinstance(oper, tc.Tensor) and isinstance(state, tc.Tensor):
        return _single_expect(oper, state)

    # Handling a list of operators
    elif isinstance(oper, (list, tc.Tensor)):
        if isinstance(state, tc.Tensor):
            return tc.tensor([_single_expect(o, state) for o in oper], dtype=tc.complex64)

    # Handling a list of states
    elif isinstance(state, (list, tc.Tensor)):
        return tc.tensor([_single_expect(oper, x) for x in state], dtype=tc.complex64)

    else:
        raise TypeError('Arguments must be torch.Tensors or lists thereof')


def density2prob(rho_in: tc.Tensor,
                 bases: Optional[Dict] = None,
                 tol: Optional[float] = None,
                 _dict: Optional[bool] = True) -> Union[Dict, np.ndarray]:
    """
    Transform density matrix into probability distribution with provided bases.

    Args:
        rho_in: density matrix;
        bases: basis set, should be input as a dict with format {'Bases': List[tc.Tensor], 'BasesName': List[str]};
        tol: probability under this threshold will not be shown;
        _dict: return a dict or np.array.
    """
    _qn = int(np.log2(rho_in.shape[0]))  # Number of qubits

    if bases is None:
        # Generate basis states
        _view_basis = [tc.zeros((2 ** _qn, 1), dtype=tc.complex64).scatter_(0, tc.tensor([[ii]]), 1) for ii in
                       range(2 ** _qn)]
        # Generate basis names
        _basis_name = [''.join(ii) for ii in itertools.product('01', repeat=_qn)]
    else:
        try:
            _view_basis, _basis_name = bases['Bases'], bases['BasesName']
        except ValueError:
            raise ValueError(
                'The input bases should be a dict with format {\'Bases\': List[tc.Tensor], \'BasesName\': List[str]}'
            )

    # Calculate probabilities
    _prob = [tc.abs(tc_expect(rho_in, base.view(-1))).item() for base in _view_basis]

    # Create dictionary and normalize
    _prob_sum = sum(_prob)
    if _dict:
        return {name: prob / _prob_sum for name, prob in zip(_basis_name, _prob) if tol is None or prob >= tol}
    else:
        return np.array(_prob) / _prob_sum


def plot_histogram(prob_psi: Union[Dict, np.ndarray],
                   threshold: Optional[float] = None,
                   title: Optional[str] = None,
                   filename: Optional[str] = None,
                   transparent: bool = False,
                   spines: bool = True,
                   show: bool = True,
                   **kwargs):
    """
    Plot a histogram of probability distribution.

    Args:
        prob_psi: probability of states, should be input as a dict or np.ndarray;
        threshold: minimum value to plot;
        title: title of the fig, while None, it does not work;
        filename: location to save the fig, while None, it does not work;
        transparent: whether to save the fig with transparent background;
        spines: whether to show the spines of the fig;
        show: whether to show the fig;
    """
    if not threshold:
        threshold = 0.
    if isinstance(prob_psi, Dict):
        qnumber = len(next(iter(prob_psi)))
        _basis_name, _prob_distribution = zip(*[(k, v) for k, v in prob_psi.items() if v >= threshold])
    elif isinstance(prob_psi, np.ndarray):
        qnumber = int(np.log2(prob_psi.shape[0]))
        _basis_name = [''.join(ii) for ii in itertools.product('01', repeat=qnumber)]
        _prob_distribution = [v for v in prob_psi if v >= threshold]
        _basis_name = [_basis_name[i] for i, v in enumerate(prob_psi) if v >= threshold]
    else:
        raise TypeError('Prob distribution should be input as a dict, or np.array.')

    title = title or f'Probability distribution qnumber={qnumber}'

    plt.figure(dpi=300, figsize=kwargs.get('figsize', (10, 8)))
    plt.bar(_basis_name, _prob_distribution,
            yerr=kwargs.get('yerr', None),
            color=kwargs.get('color', 'b'))
    plt.ylim(ymin=0, ymax=kwargs.get('ymax', 1.))
    plt.xticks(rotation=-45, fontsize=kwargs.get('xticks_fontsize', 22))
    plt.yticks(fontsize=kwargs.get('yticks_fontsize', 22))
    if kwargs.get('yticks'):
        plt.yticks(ticks=kwargs.get('yticks'))
    plt.title(title, fontsize=kwargs.get('title_fontsize', 27))
    plt.xlabel('Bitstring', fontsize=kwargs.get('xlabel_fontsize', 24))
    plt.ylabel('Probability', fontsize=kwargs.get('ylabel_fontsize', 24))
    plt.tight_layout()

    if not spines:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    if filename:
        plt.savefig(filename, transparent=transparent, dpi=300)

    if show:
        plt.show()


def plot_histogram_multiBars(
        prob_psi1: Union[Dict, np.ndarray],
        prob_psi2: Union[Dict, np.ndarray],
        prob_psi3: Union[Dict, np.ndarray] = None,
        prob_psi4: Union[Dict, np.ndarray] = None,
        labels: list[str] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        transparent: bool = False,
        spines: bool = True,
        show: bool = True,
        threshold: float = None,
        colors: Optional[List[str]] = None,
        text: Optional[str] = None,
        **kwargs
):
    """
    Plot a histogram of probability distribution for up to four datasets,
    omitting bars where all datasets have values below a threshold.

    Args:
        prob_psi1: First probability distribution, as a dict or np.ndarray;
        prob_psi2: Second probability distribution, as a dict or np.ndarray;
        prob_psi3: Third probability distribution, optional, as a dict or np.ndarray;
        prob_psi4: Fourth probability distribution, optional, as a dict or np.ndarray;
        labels: List of labels for the datasets;
        title: Title of the figure, defaults to a generic title if None;
        filename: Location to save the figure, does nothing if None;
        transparent: Whether to save the figure with a transparent background;
        spines: Whether to show the spines of the figure;
        show: Whether to show the figure;
        threshold: Value below which bars are not shown if all datasets are below it;
        colors: List of colors for each dataset bars;
        text: Text to be shown if all datasets are below it.
    """
    if colors is None:
        colors = ['b', 'orange', 'g', 'r']

    def extract_data(prob_psi):
        if isinstance(prob_psi, Dict):
            basis_name = list(prob_psi.keys())
            prob_distribution = list(prob_psi.values())
        elif isinstance(prob_psi, np.ndarray):
            qnumber = int(np.log2(prob_psi.shape[0]))
            prob_distribution = prob_psi
            basis_name = [''.join(ii) for ii in itertools.product('01', repeat=qnumber)]
        else:
            raise TypeError('Prob distribution should be input as a dict or np.array.')
        return basis_name, prob_distribution

    if not threshold:
        threshold = 0.

    datasets = [prob_psi1, prob_psi2, prob_psi3, prob_psi4]
    distributions = [extract_data(data) for data in datasets if data is not None]

    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(distributions))]

    basis_names = [d[0] for d in distributions]
    prob_distributions = [d[1] for d in distributions]

    # Filter out indices where all datasets are below the threshold
    filtered_indices = [i for i in range(len(basis_names[0]))
                        if any(prob[i] >= threshold for prob in prob_distributions)]

    basis_name_filtered = [basis_names[0][i] for i in filtered_indices]
    prob_distributions_filtered = [[prob[i] for i in filtered_indices] for prob in prob_distributions]

    title = title or 'Probability Distribution'

    plt.figure(dpi=300, figsize=kwargs.get('figsize', (10, 8)))
    n_bars = len(distributions)
    group_width = 0.8
    bar_width = group_width / n_bars
    x = np.arange(len(basis_name_filtered))

    for i, (prob_distribution, color) in enumerate(zip(prob_distributions_filtered, colors[:n_bars])):
        bar_positions = x - (group_width - bar_width) / 2 + i * bar_width
        plt.bar(bar_positions, prob_distribution, bar_width, label=labels[i], color=color)

    plt.ylim(ymin=0, ymax=kwargs.get('ymax', 1))
    plt.xticks(x, basis_name_filtered, rotation=-45, fontsize=kwargs.get('xticks_fontsize', 22))
    plt.yticks(fontsize=kwargs.get('yticks_fontsize', 22))
    plt.title(title, fontsize=kwargs.get('title_fontsize', 27))
    plt.xlabel('Bitstring', fontsize=kwargs.get('xlabel_fontsize', 24))
    plt.ylabel('Probability', fontsize=kwargs.get('ylabel_fontsize', 24))
    plt.legend(fontsize=kwargs.get('legend_fontsize', 22))
    plt.tight_layout()

    if kwargs.get('yticks'):
        plt.yticks(kwargs.get('yticks'))

    if not spines:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    if text:
        if kwargs.get('text_loc'):
            _x, _y = kwargs.get('text_loc')
        else:
            _x, _y = (0, 0)
        plt.text(_x, _y, text, ha='center', va='center', fontsize=kwargs.get('text_fontsize', 22),
                 bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.5'),
                 fontdict=kwargs.get('text_fontdict', {}))

    if filename:
        plt.savefig(filename, transparent=transparent, dpi=300)

    if show:
        plt.show()


def select_device(device: Optional[Union[str, int]] = None):
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


def gates_list(N: int, basis_gates: Optional[List[str]] = None) -> List[str]:
    """
    Generates a series of gate sets as basis.

    For N qubits, it creates all possible combinations of the specified basis gates.
    Example:
    - N = 1 --> ['I', 'X', 'Y', 'Z']
    - N = 2 --> ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']

    Args:
        N: Number of qubits.
        basis_gates: List of basis gates. Default is ['I', 'X', 'Y', 'Z'].

    Returns:
        List of strings representing gate combinations.
    """
    if basis_gates is None:
        basis_gates = ['I', 'X', 'Y', 'Z']

    return [''.join(gates) for gates in itertools.product(basis_gates, repeat=N)]


def name2matrix(operation_name: str, dtype=tc.complex64, device: Union[str, int] = 'cpu'):
    """
    Generate a product matrix for a given operation sequence.

    Args:
        operation_name: String representing operations, like 'ZZZ'.
        dtype: Data type of the tensors.
        device: Computation device ('cpu' or GPU index).

    Returns:
        Product matrix corresponding to the sequence of operations.
    """
    # Define the operation matrices
    operations = {
        'I': tc.eye(2, dtype=dtype, device=device),
        'X': tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
        'Y': -1j * tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        'Z': tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    }

    # Generate the list of operation matrices
    operation_list = [operations[letter] for letter in operation_name]

    # Compute the tensor product of the matrices
    return reduce(tc.kron, operation_list)


def sqrt_matrix(density_matrix: tc.Tensor) -> tc.Tensor:
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


def validDensityMatrix(rho,
                       methodIdx: int = 1,
                       constraints: str = 'eq',
                       hermitian: bool = True,
                       device: Union[str, int] = 'cpu'):
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
    bounds = [(0.0, traceV + 0.001) for _ in range(len(ps))]

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

    psi, newPs = tc.tensor(psi), tc.tensor(newPs, dtype=tc.complex64)

    rho_semi = psi @ np.diag(newPs) @ psi.T.conj()
    return rho_semi.to(device=device)


def iSing_hamiltonian_near(qnumber: int) -> tc.Tensor:
    # Predefined matrices
    qeye = tc.tensor([[1, 0], [0, 1]], dtype=tc.complex64)
    sigma_z = tc.tensor([[1, 0], [0, -1]], dtype=tc.complex64)

    def tensor_sigmaz(N: int, order: int):
        _list = [sigma_z if _i == order else qeye for _i in range(N)]
        return reduce(tc.kron, _list)

    # Initialize Hamiltonian
    Ham = tc.zeros(size=(2 ** qnumber, 2 ** qnumber), dtype=tc.complex64)

    # Construct Hamiltonian
    for i in range(qnumber - 1):
        Ham -= tc.matmul(tensor_sigmaz(qnumber, i), tensor_sigmaz(qnumber, i + 1))

    return Ham
