"""
Author: weiguo_ma
Time: 04.26.2023
Contact: weiguo.m@iphy.ac.cn
"""
from copy import deepcopy
from typing import Optional, Dict, Union, List, Any, Tuple

import tensornetwork as tn
import torch as tc

from Library.AbstractCircuit import QuantumCircuit
from Library.NoiseChannel import NoiseChannel
from Library.QuantumGates.AbstractGate import QuantumGate
from Library.TNNOptimizer import svd_right2left, qr_left2right, svdKappa_left2right, cal_entropy, checkConnectivity
from Library.tools import select_device, EdgeName2AxisName


class TensorCircuit(QuantumCircuit):
    def __init__(self, qn: int,
                 ideal: bool = True,
                 noiseType: str = 'no',
                 chiFileDict: Optional[Dict[str, Dict[str, Any]]] = None,
                 crossTalk: bool = False,
                 chi: Optional[int] = None, kappa: Optional[int] = None,
                 tnn_optimize: bool = True,
                 chip: Optional[str] = None, device: Optional[Union[str, int]] = None,
                 _entropy: bool = False):
        """
        Args:
            ideal: Whether the circuit is ideal.  --> Whether the one-qubit gate is ideal or not.
            noiseType: The type of the noise channel.   --> Which kind of noise channel is added to the two-qubit gate.
            chiFileDict: The filename of the chi matrix.
            chi: The chi parameter of the TNN.
            kappa: The kappa parameter of the TNN.
            tnn_optimize: Whether the TNN is optimized.
            chip: The name of the chip.
            device: The device of the torch;
            _entropy: Whether to record the system entropy.
        """
        self.realNoise = True if noiseType == 'realNoise' and not ideal else False
        self.device = select_device(device)
        self.dtype = tc.complex64

        super(TensorCircuit, self).__init__(self.realNoise,
                                            noiseFiles=chiFileDict, dtype=self.dtype, device=self.device)

        # About program
        self.fVR = None
        self.qnumber = qn
        self.state = None
        self.initState = None

        # Entropy
        self._entropy = _entropy
        self._entropyList = {f'qEntropy_{_i}': [] for _i in range(self.qnumber)}
        self._entropyList.update({f'dimension_{_i}': [] for _i in range(self.qnumber)})

        # Noisy Circuit Setting
        self.ideal = ideal
        self.idealNoise = False
        self.unified = None
        self.crossTalk = crossTalk

        if not self.ideal:
            self.Noise = NoiseChannel(chip=chip, device=self.device)
            if noiseType == 'unified':
                self.unified = True
            elif noiseType == 'realNoise':
                self.realNoise, self.crossTalk = True, False
            elif noiseType == 'idealNoise':
                self.idealNoise = True
            else:
                raise TypeError(f'Noise type "{noiseType}" is not supported yet.')

        # Density Matrix
        self.DM = None
        self.DMNode = None

        # TNN Truncation
        self.chi = chi
        self.kappa = kappa
        self.tnn_optimize = tnn_optimize

    @staticmethod
    def _getBond(target_idx: int, idxList: Union[List[int], range], Name: List[str], Letters: str, _left: bool) \
            -> List[str]:
        return [
            Letters[_idx_ - target_idx - 1] if not _left and f'bond_{target_idx}_{_idx_}' in Name else
            Letters[_idx_] if _left and f'bond_{_idx_}_{target_idx}' in Name else ''
            for _idx_ in idxList
        ]

    @staticmethod
    def _calOrder(minIdx: int, maxIdx: Optional[int] = None, lBond: List = None, rBond: List = None,
                  smallI: bool = False, bigI: bool = False, single: bool = False):
        if single:
            _lNames = [f'bond_{i}_{minIdx}' for i, item in enumerate(lBond) if item] if lBond else []
            _rNames = [f'bond_{minIdx}_{i + minIdx + 1}' for i, item in enumerate(rBond) if item] if rBond else []
            # lBond, phy, I, rBond
            _return = _lNames + [f'physics_{minIdx}'] + [f'I_{minIdx}'] * smallI + _rNames
        else:
            # Type-in lBond and rBond should be (minLBond, maxLBond), (minRBond, maxRBond)
            _lNames = [f'bond_{i}_{minIdx}' for i, item in enumerate(lBond[0]) if item] if lBond else []
            _lNames.extend([f'bond_{i}_{maxIdx}' for i, item in enumerate(lBond[1]) if item] if lBond else [])

            _rNames = [f'bond_{minIdx}_{i+minIdx+1}' for i, item in enumerate(rBond[0]) if item] if rBond else []
            _rNames.extend([f'bond_{maxIdx}_{i+maxIdx+1}' for i, item in enumerate(rBond[1]) if item] if rBond else [])
            # minLBond, maxLBond, minPhy, maxPhy, minI, maxI, minRBond, maxRBond
            _return = (
                    _lNames + [f'physics_{minIdx}', f'physics_{maxIdx}'] +
                    [f'I_{minIdx}'] * smallI + [f'I_{maxIdx}'] * bigI + _rNames
            )
        return _return

    def _add_gate(self, _qubits: List[tn.AbstractNode], _layer_num: int, _oqs: List[int]):
        r"""
        Add quantum Gate to tensornetwork

        Args:
            _oqs: operating qubits.

        Returns:
            Tensornetwork after adding gate.
        """

        gate = self.layers[_layer_num]
        if not gate:        # Stopping condition
            return None
        _maxIdx, _minIdx = max(_oqs), min(_oqs)

        if not isinstance(_qubits, List):
            raise TypeError('Qubit must be a list of nodes.')
        if not isinstance(gate, QuantumGate):
            raise TypeError(f'Gate must be a TensorGate, current type is {type(gate)}.')
        if not isinstance(_oqs, List):
            raise TypeError('Operating qubits must be a list.')
        if _maxIdx >= self.qnumber:
            raise ValueError(f'Qubit index out of range, max index is Q{_maxIdx}.')

        single = gate.single

        if not single:  # Two-qubit gate
            """
                           | i  | j                         | w              | p
                        ___|____|___                    ____|____        ____|____
                        |          |                    |       |        |       |
                        |    DG    |---- q        l ----| Qubit |--------| Qubit |---- r
                        |__________|                    |_______|    k   |_______|
                           |    |                           |                |
                           | w  | p                         | m              | n
            """
            if len(_oqs) != 2 or _minIdx == _maxIdx:
                raise ValueError('Invalid operating qubits for a two-qubit gate.')

            _gateTensor = gate.tensor

            if self.idealNoise and not gate.ideal:
                # Adding depolarization noise channel to Two-qubit gate
                _gateTensor = tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2, _gateTensor)

            # Created a new node in memory
            _contract_qubits = tn.contract_between(_qubits[_minIdx], _qubits[_maxIdx],
                                                   name=f'q_{_oqs[0]}_{_oqs[1]}',
                                                   allow_outer_product=True)
            # ProcessFunction, for details, see the function definition.
            EdgeName2AxisName([_contract_qubits])

            _contract_qubitsAxisName = _contract_qubits.axis_names

            _gString = 'ijwpq' if len(_gateTensor.shape) == 5 else 'ijwp'
            if _oqs[0] > _oqs[1]:
                _gString = _gString.replace('wp', 'pw')

            _smallI, _bigI = f'I_{min(_oqs)}' in _contract_qubitsAxisName, f'I_{max(_oqs)}' in _contract_qubitsAxisName

            _minL_Idx, _minR_Idx, _maxL_Idx, _maxR_Idx = (
                range(_minIdx), range(_minIdx + 1, self.qnumber),
                range(_maxIdx), range(_maxIdx + 1, self.qnumber)
            )
            _min_lList, _min_rList, _max_lList, _max_rList = 'labch', 'defg', 'stuz', 'rvwxy'  # HARD CODE

            _min_lBond, _min_rBond, _max_lBond, _max_rBond = (
                self._getBond(_minIdx, _minL_Idx, _contract_qubitsAxisName, _min_lList, True),
                self._getBond(_minIdx, _minR_Idx, _contract_qubitsAxisName, _min_rList, False),
                self._getBond(_maxIdx, _maxL_Idx, _contract_qubitsAxisName, _max_lList, True),
                self._getBond(_maxIdx, _maxR_Idx, _contract_qubitsAxisName, _max_rList, False)
            )

            _qString = ''.join(_min_lBond + _max_lBond + ['wp', 'm' * _smallI, 'n' * _bigI] + _min_rBond + _max_rBond)

            _qAFString = _qString.replace('wp', _gString[:2] + _gString[-1] * (len(_gateTensor.shape) == 5))
            if _oqs[0] > _oqs[1]:
                _qAFString = _qAFString.replace('ij', 'ji')

            _reorderAxisName = self._calOrder(_minIdx, _maxIdx, lBond=[_min_lBond, _max_lBond],
                                              rBond=[_min_rBond, _max_rBond], smallI=_smallI, bigI=_bigI)

            _contract_qubits.reorder_edges([_contract_qubits[_element] for _element in _reorderAxisName])
            _contract_qubitsTensor_AoP = tc.einsum(f'{_gString}, {_qString} -> {_qAFString}',
                                                   _gateTensor, _contract_qubits.tensor)
            _qShape = _contract_qubitsTensor_AoP.shape

            _mIdx, _qIdx = _qAFString.find('m'), _qAFString.find('n')

            if not gate.ideal:
                if _mIdx != -1:
                    _qShape = _qShape[:_mIdx - 1] + (_qShape[_mIdx - 1] * _qShape[_mIdx],) + _qShape[_mIdx + 1:]
                    _contract_qubits.set_tensor(tc.reshape(_contract_qubitsTensor_AoP, _qShape))
                else:
                    _AFStringT = f'{_qAFString.replace("q", "")}{"q"}'
                    _contract_qubits.set_tensor(
                        tc.einsum(f"{_qAFString} -> {_AFStringT}", _contract_qubitsTensor_AoP))

                    _reorderAxisName.append(f'I_{_minIdx}')
                    _contract_qubits.add_axis_names(_reorderAxisName)
                    _new_edge = tn.Edge(_contract_qubits,
                                        axis1=len(_contract_qubits.edges), name=f'I_{_minIdx}')
                    _contract_qubits.edges.append(_new_edge)
                    _contract_qubits.add_edge(_new_edge, f'I_{_minIdx}')

                    _smallI = True
                    _reorderAxisName = self._calOrder(_minIdx, _maxIdx, lBond=[_min_lBond, _max_lBond],
                                                      rBond=[_min_rBond, _max_rBond], smallI=_smallI, bigI=_bigI)
                    _contract_qubits.reorder_edges([_contract_qubits[_element] for _element in _reorderAxisName])
            else:
                _contract_qubits.set_tensor(_contract_qubitsTensor_AoP)

            # Split back to two qubits
            _left_AxisName = (
                    [f'bond_{_i}_{_minIdx}' for _i, _item in enumerate(_min_lBond) if _item] +
                    [f'physics_{_minIdx}'] +
                    [f'I_{_minIdx}'] * ((self.idealNoise or self.realNoise) and _smallI) +
                    [f'bond_{_minIdx}_{_minIdx + 1 + _i}' for _i, _item in enumerate(_min_rBond) if _item]
            )

            _right_AxisName = (
                    [f'bond_{_i}_{_maxIdx}' for _i, _item in enumerate(_max_lBond) if _item] +
                    [f'physics_{_maxIdx}'] +
                    [f'I_{_maxIdx}'] * ((self.idealNoise or self.realNoise) and _bigI) +
                    [f'bond_{_maxIdx}_{_maxIdx + 1 + _i}' for _i, _item in enumerate(_max_rBond) if _item]
            )
            _left_edges, _right_edges = (
                [_contract_qubits[name] for name in _left_AxisName],
                [_contract_qubits[name] for name in _right_AxisName]
            )

            _qubits[_minIdx], _qubits[_maxIdx], _ = tn.split_node(_contract_qubits,
                                                                  left_edges=_left_edges,
                                                                  right_edges=_right_edges,
                                                                  left_name=f'qubit_{_minIdx}',
                                                                  right_name=f'qubit_{_maxIdx}',
                                                                  edge_name=f'bond_{_minIdx}_{_maxIdx}')
            EdgeName2AxisName([_qubits[_minIdx], _qubits[_maxIdx]])
        else:
            """
                            | m                             | i
                        ____|___                        ____|____
                        |      |                        |       |   
                        |  SG  |---- n            l ----| Qubit |---- r
                        |______|                        |_______|
                            |                               |
                            | i                             | j
            """
            gate_list = [
                tc.reshape(
                    tc.einsum('nlm, ljk, ji -> nimk', self.Noise.dpCTensor, self.Noise.apdeCTensor, gate.tensor),
                    (2, 2, -1))
                if (self.idealNoise or self.unified) and not gate.ideal else gate.tensor
                for _idx in _oqs
            ]
            for _i, _bit in enumerate(_oqs):
                _qubit, _gShape = _qubits[_bit], gate_list[_i].shape
                _qubitAxisName, _qubitShape = _qubit.axis_names, _qubit.tensor.shape

                _I = f'I_{_bit}' in _qubitAxisName

                _lList, _rList = 'labcdef', 'ropqxyz'  # HARD CODE
                _lIdx, _rIdx = range(_bit), range(_bit + 1, self.qnumber)
                _lBond, _rBond = (
                    self._getBond(_bit, _lIdx, _qubitAxisName, _lList, True),
                    self._getBond(_bit, _rIdx, _qubitAxisName, _rList, False)
                )

                _qString = ''.join(_lBond + ['i', 'j' * _I] + _rBond)
                _gString = 'min' if len(_gShape) == 3 else 'mi'
                _qAFString = _qString.replace('i', 'mn') if _gString == 'min' else _qString.replace('i', 'm')

                _reorderAxisName = self._calOrder(minIdx=_bit, lBond=_lBond, smallI=_I, rBond=_rBond, single=True)

                if _qubitAxisName != _reorderAxisName:
                    _qubit.reorder_edges([_qubit[_element] for _element in _reorderAxisName])
                _qubitTensor_AOP = tc.einsum(f'{_gString}, {_qString} -> {_qAFString}', gate_list[_i], _qubit.tensor)
                _qShape = _qubitTensor_AOP.shape

                _jIdx, _nIdx = _qAFString.find('j'), _qAFString.find('n')
                if _jIdx != -1:  # Exist j -> Noisy Qubit
                    if _nIdx != -1:
                        _qShape = _qShape[:_jIdx - 1] + (_qShape[_jIdx - 1] * _qShape[_jIdx],) + _qShape[_jIdx + 1:]
                        _qubitTensor_AOP = tc.reshape(_qubitTensor_AOP, _qShape)
                    _qubit.set_tensor(_qubitTensor_AOP)
                else:
                    if 'n' in _gString:
                        _AFStringT = f'{_qAFString.replace("n", "")}{"n"}'
                        _qubit.set_tensor(
                            tc.einsum(f"{_qAFString} -> {_AFStringT}", _qubitTensor_AOP))

                        _qubitAxisName.insert(_qubitAxisName.index(f'physics_{_bit}') + 1, f'I_{_bit}')
                        _qubit.add_axis_names(_qubitAxisName)
                        _new_edge = tn.Edge(_qubit, axis1=len(_qubit.edges), name=f'I_{_bit}')
                        _qubit.edges.insert(_qubitAxisName.index(f'I_{_bit}'), _new_edge)
                        _qubit.add_edge(_new_edge, f'I_{_bit}')

                        _I = True
                        _reorderAxisName = self._calOrder(minIdx=_bit, lBond=_lBond, smallI=_I, rBond=_rBond,
                                                          single=True)
                        _qubit.reorder_edges([_qubit[_element] for _element in _reorderAxisName])
                    else:  # Ideal Gate and Qubit
                        _qubit.set_tensor(_qubitTensor_AOP)

    def _calculate_DM(self, state_vector: bool = False,
                      reduced_index: Optional[List[Union[List[int], int]]] = None) -> tc.Tensor:
        """
        Calculate the density matrix of the state.

        Args:
            state_vector: if True, the state is a state vector, otherwise, the state is a density matrix.
            reduced_index: the state[index] to be reduced, which means the physics_con-physics of sub-density matrix
                                will be connected and contracted.

        Returns:
            _dm: the density matrix node;
            _dm_tensor: the density matrix tensor.

        Additional information:
            A mixed state(noisy situation) is generally cannot be efficiently represented by a state vector but DM.

        Raises:
            ValueError: if the state is chosen to be a vector but is noisy;
            ValueError: Reduced index should be empty as [] or None.
        """
        if not reduced_index:
            reduced_index = None

        if reduced_index and max(reduced_index) >= self.qnumber:
            raise ValueError('Reduced index should not be larger than the qubit number.')

        if not state_vector:
            _qubits_conj = deepcopy(self.state)  # Node.copy() func. cannot work correctly
            for _qubit in _qubits_conj:
                _qubit.set_tensor(_qubit.tensor.conj())

            # Differential name the conjugate qubits' edges name to permute the order of the indices
            for _i, _qubit_conj in enumerate(_qubits_conj):
                _qubit_conj.set_name(f'con_{_qubit_conj.name}')
                for _ii, _edge in enumerate(_qubit_conj.edges):
                    if 'physics' in _edge.name:
                        _edge.set_name(f'con_{_qubit_conj[_ii].name}')
                        _qubit_conj.axis_names[_ii] = f'con_{_qubit_conj.axis_names[_ii]}'

            for i in range(len(self.state)):
                if not self.ideal and f'I_{i}' in self.state[i].axis_names:
                    tn.connect(self.state[i][f'I_{i}'], _qubits_conj[i][f'I_{i}'])

            # Reduced density matrix
            if reduced_index:
                reduced_index = [reduced_index] if isinstance(reduced_index, int) else reduced_index
                if not isinstance(reduced_index, list):
                    raise TypeError('reduced_index should be int or list[int]')
                for _idx in reduced_index:
                    tn.connect(self.state[_idx][f'physics_{_idx}'], _qubits_conj[_idx][f'con_physics_{_idx}'])
            else:
                reduced_index = []

            _numList = [_i for _i in range(self.qnumber) if _i not in reduced_index]
            _qOutOrder, _conQOutOrder = (
                [self.state[i][f'physics_{i}'] for i in _numList],
                [_qubits_conj[i][f'con_physics_{i}'] for i in _numList]
            )

            _dm = tn.contractors.auto(self.state + _qubits_conj,
                                      output_edge_order=_qOutOrder + _conQOutOrder)

            _reshape_size = self.qnumber - len(reduced_index)

            self.DM, self.DMNode = _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size)), self.DMNode
            return self.DM
        else:
            if not self.fVR and not self.ideal:
                raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')
            if reduced_index is not None:
                raise ValueError('State vector cannot efficiently represents the reduced density matrix.')

            _outOrder = [self.state[i][f'physics_{i}'] for i in list(range(self.qnumber))]
            _vector = tn.contractors.auto(self.state, output_edge_order=_outOrder)

            self.DM = _vector.tensor.reshape((2 ** self.qnumber, 1)) if not self.fVR else _vector.tensor
            return self.DM

    def forward(self,
                _state: List[tn.AbstractNode],
                state_vector: bool = False,
                reduced_index: Optional[List] = None,
                forceVectorRequire: bool = False,
                _require_nodes: bool = False) -> Union[tc.Tensor, Dict, Tuple]:
        """
        Forward propagation of tensornetwork.

        Returns:
            self.state: tensornetwork after forward propagation.
        """
        self.initState, self.qnumber, self.fVR = _state, len(_state), forceVectorRequire

        for _i, layer in enumerate(self.layers):
            self._add_gate(_state, _i, _oqs=self._oqs_list[_i])
            self.Truncate = True if layer is None else False
            #
            if self.Truncate and self.tnn_optimize:
                if not self.ideal:
                    svdKappa_left2right(_state, kappa=self.kappa)
                if checkConnectivity(_state):
                    qr_left2right(_state)
                    svd_right2left(_state, chi=self.chi)
            if self._entropy and not self.Truncate:
                _entropy = cal_entropy(_state)
                for _ii in range(len(_state)):
                    self._entropyList[f'dimension_{_ii}'].append(_entropy[f'dimension_{_ii}'])
                    self._entropyList[f'qEntropy_{_ii}'].append(_entropy[f'qEntropy_{_ii}'])
            #
            if self.Truncate:
                self.Truncate = False
        # LastLayer noise-truncation
        if self.tnn_optimize and not self.ideal:
            svdKappa_left2right(_state, kappa=self.kappa)

        self.state = _state
        _nodes = deepcopy(_state) if _require_nodes else None
        _dm = self._calculate_DM(state_vector=state_vector, reduced_index=reduced_index)

        return (_nodes, _dm) if _require_nodes else _dm
