"""
Author: weiguo_ma
Time: 06.21.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import tensornetwork as tn
from typing import List


tn.set_default_backend("pytorch")


def superoperator_to_kraus(superoperator: tc.Tensor, atol: float = 1e-10) -> List[tc.Tensor]:
	r"""
    Returns a Kraus representation of a channel specified via the superoperator matrix.

    Quantum channel E: L(H1) -> L(H2) may be described by a collection of operators A_i, called
    Kraus operators, such that

        $$
        E(\rho) = \sum_i A_i \rho A_i^\dagger.
        $$

    Kraus representation is not unique. Alternatively, E may be specified by its superoperator
    matrix K(E) defined so that

        $$
        K(E) vec(\rho) = vec(E(\rho))
        $$

    where the vectorization map $vec$ rearranges d-by-d matrices into d**2-dimensional vectors.
    Superoperator matrix is unique for a given channel. It is also called the natural
    representation of a quantum channel.

    The most expensive step in the computation of a Kraus representation from a superoperator
    matrix is eigendecomposition. Therefore, the cost of the conversion is O(d**6) where d is
    the dimension of the input and output Hilbert space.

    Args:
        superoperator: Superoperator matrix specifying a quantum channel.
        atol: Tolerance used to check which Kraus operators to omit.

    Returns:
        Sequence of Kraus operators of the channel specified by superoperator.
        Kraus operators with Frobenius norm smaller than atol are omitted.

    Raises:
        ValueError: If superoperator is not a valid superoperator matrix.
    """
	return choi_to_kraus(superoperator_to_choi(superoperator), atol=atol)


def choi_to_kraus(choi: tc.Tensor, atol: float = 1e-10) -> List[tc.Tensor]:
	r"""Returns a Kraus representation of a channel with given Choi matrix.

	Quantum channel E: L(H1) -> L(H2) may be described by a collection of operators A_i, called
	Kraus operators, such that

		$$
		E(\rho) = \sum_i A_i \rho A_i^\dagger.
		$$

	Kraus representation is not unique. Alternatively, E may be specified by its Choi matrix J(E)
	defined as

		$$
		J(E) = (E \otimes I)(|\phi\rangle\langle\phi|)
		$$

	where $|\phi\rangle = \sum_i|i\rangle|i\rangle$ is the unnormalized maximally entangled state
	and I: L(H1) -> L(H1) is the identity map. Choi matrix is unique for a given channel.

	The most expensive step in the computation of a Kraus representation from a Choi matrix is
	the eigendecomposition of the Choi. Therefore, the cost of the conversion is O(d**6) where
	d is the dimension of the input and output Hilbert space.

	Args:
		choi: Choi matrix of the channel.
		atol: Tolerance used in checking if choi is positive and in deciding which Kraus
			operators to omit.
	Returns:
			Approximate Kraus representation of the quantum channel specified via a Choi matrix.
			Kraus operators with Frobenius norm smaller than atol are omitted.

		Raises:
			ValueError: when choi is not a positive square matrix.
		"""
	d = int(tc.sqrt(tc.tensor(choi.shape[0])))
	if choi.shape != (d * d, d * d):
		raise ValueError(f"Invalid Choi matrix shape, expected {(d * d, d * d)}, got {choi.shape}")
	if not tc.allclose(choi, choi.T.conj(), atol=atol):
		raise ValueError("Choi matrix must be Hermitian")

	w, v = tc.linalg.eigh(choi)
	if tc.any(w < -atol):
		raise ValueError(f"Choi matrix must be positive, got one with eigenvalues {w}")

	w = tc.maximum(w.to(dtype=tc.float64), tc.tensor(0.0, dtype=tc.float64)).to(dtype=tc.complex128)
	u = tc.sqrt(w) * v
	keep = tc.norm(u, dim=-1) > atol
	return [k.reshape(d, d) for k, keep_i in zip(u.T, keep) if keep_i]


def superoperator_to_choi(superoperator: tc.Tensor) -> tc.Tensor:
	r"""Returns the Choi matrix of a quantum channel specified via the superoperator matrix.

	Quantum channel E: L(H1) -> L(H2) may be specified by its Choi matrix J(E) defined as

		$$
		J(E) = (E \otimes I)(|\phi\rangle\langle\phi|)
		$$

	where $|\phi\rangle = \sum_i|i\rangle|i\rangle$ is the unnormalized maximally entangled state
	and I: L(H1) -> L(H1) is the identity map. Choi matrix is unique for a given channel.
	Alternatively, E may be specified by its superoperator matrix K(E) defined so that

		$$
		K(E) vec(\rho) = vec(E(\rho))
		$$

	where the vectorization map $vec$ rearranges d-by-d matrices into d**2-dimensional vectors.
	Superoperator matrix is unique for a given channel. It is also called the natural
	representation of a quantum channel.

	A quantum channel can be viewed as a tensor with four indices. Different ways of grouping
	the indices into two pairs yield different matrix representations of the channel, including
	the superoperator and Choi representations. Hence, the conversion between the superoperator
	and Choi matrices is a permutation of matrix elements effected by reshaping the array and
	swapping its axes. Therefore, its cost is O(d**4) where d is the dimension of the input and
	output Hilbert space.

	Args:
		superoperator: Superoperator matrix specifying a quantum channel.

	Returns:
		Choi matrix of the channel specified by superoperator.

	Raises:
		ValueError: If superoperator has invalid shape.
	"""
	d = int(tc.sqrt(tc.tensor(superoperator.shape[0])))
	if superoperator.shape != (d * d, d * d):
		raise ValueError(
			f"Invalid superoperator matrix shape, expected {(d * d, d * d)}, "
			f"got {superoperator.shape}"
		)

	s = superoperator.reshape(d, d, d, d)
	c = s.permute(0, 2, 1, 3)
	return c.reshape(d * d, d * d)


def kraus_to_superoperator(kraus_operators: List[tc.Tensor]) -> tc.Tensor:
	r"""Returns the matrix representation of the linear map with given Kraus operators.

	Quantum channel E: L(H1) -> L(H2) may be described by a collection of operators A_i, called
	Kraus operators, such that

		$$
		E(\rho) = \sum_i A_i \rho A_i^\dagger.
		$$

	Kraus representation is not unique. Alternatively, E may be specified by its superoperator
	matrix K(E) defined so that

		$$
		K(E) vec(\rho) = vec(E(\rho))
		$$

	where the vectorization map $vec$ rearranges d-by-d matrices into d**2-dimensional vectors.
	Superoperator matrix is unique for a given channel. It is also called the natural
	representation of a quantum channel.

	The computation of the superoperator matrix from a Kraus representation involves the sum of
	Kronecker products of all Kraus operators. This has the cost of O(kd**4) where k is the number
	of Kraus operators and d is the dimension of the input and output Hilbert space.

	Args:
		kraus_operators: Sequence of Kraus operators specifying a quantum channel.

	Returns:
		Superoperator matrix of the channel specified by kraus_operators.
	"""
	d_out, d_in = kraus_operators[0].shape
	ops_list = tc.stack(kraus_operators)
	m = tc.einsum('bij,bkl->ikjl', ops_list, ops_list.conj())
	return m.reshape((d_out * d_out, d_in * d_in))