"""
Author: weiguo_ma
Time: 04.20.2023
Contact: weiguo.m@iphy.ac.cn
"""
import numpy as np
import torch as tc


def tensorDot(*args) -> tc.Tensor:
    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    elif len(args) == 1 and isinstance(args[0], tc.Tensor):
        # tensor is called with a single Qobj as an argument, do nothing
        return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args

    if not all([isinstance(q, tc.Tensor) for q in qlist]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a torch tensor")

    out = tc.Tensor()
    for n, q in enumerate(qlist):
        if n == 0:
            out = q
        else:
            out = tc.kron(out, q)
    return out

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
        _result = tc.matmul(state.T.conj(), tc.matmul(operator, state))
        return tc.abs(_result)
        # while in physics fact, it should be real, but in computation, it is complex for technical reason.

def tc_basis(dimension: int, N: int = 0):
    ls = np.array([0] * dimension).reshape(dimension, 1)
    ls[N] = 1
    return tc.tensor(ls, dtype=tc.complex64)