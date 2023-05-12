"""
Author: weiguo_ma
Time: 04.30.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn

from Library.tools import create_ket0Series, density2prob, plot_histogram
from Library.ADCircuits import TensorCircuit
from Library.AbstractGate import AbstractGate

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # Bugs fixed in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

qnumber = 5

circuit = TensorCircuit(ideal=False, tnn_optimize=True, realNoise=True,
                        chiFilename='./data/chi/chi1.mat', chi=4, kappa=None)

circuit.add_gate(AbstractGate().h(), [0])
circuit.add_gate(AbstractGate().cnot(), [0, 1])
circuit.add_gate(AbstractGate().cnot(), [1, 2])
circuit.add_gate(AbstractGate().cnot(), [2, 3])
circuit.add_gate(AbstractGate().cnot(), [3, 4])
print(circuit)

InitState = create_ket0Series(qnumber)
state = circuit(InitState, state_vector=False, reduced_index=[], forceVectorRequire=False)
# print(state)

# Calculate probability distribution
prob_dict = density2prob(state, tolerant=None)
print(prob_dict)

# plot probability distribution
plot_histogram(prob_dict, title=None)
