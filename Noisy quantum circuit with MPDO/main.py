"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
import Library.tools as tools
from Library.ADCircuits import TensorCircuit
from Library.AbstractGate import AbstractGate

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # Bugs fixed in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

# Basic information of circuit
qnumber = 4

# Establish a quantum circuit
circuit = TensorCircuit(ideal=False)
# layer1
circuit.add_gate(AbstractGate().h(), [0, 2])
circuit.add_gate(AbstractGate().x(), [1])
# layer2
circuit.add_gate(AbstractGate().cnot(), [0, 1])
circuit.add_gate(AbstractGate().cnot(), [2, 3])
# layer3
circuit.add_gate(AbstractGate().cnot(), [1, 2])
# layer4
circuit.add_gate(AbstractGate().x(), [0, 2, 3])
circuit.add_gate(AbstractGate().h(), [1])

# Generate an initial quantum state
state = tools.create_ket0Series(qnumber)
state = circuit(state, state_vector=False, reduced_index=[0])

print(state)