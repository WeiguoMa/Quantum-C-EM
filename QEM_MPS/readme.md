# Quantum Error Mitigation via Matrix Product Operator (MPO)

This project is mainly developed from paper,

*[1] Y. Guo and S. Yang, Quantum Error Mitigation via Matrix Product Operators, PRX Quantum 3, 040313 (2022).*

The whole error mitigation process can be divided into three parts,

1. Utilize high-performance quantum process tomography (QPT) to get the super-operator of the noisy
   quantum process as matrix product operator (MPO);
2. Find the inverse of the quantum process MPO;
3. Apply the inverse MPO to the noisy quantum circuit to mitigate the error, the inverse MPO need to be
   decomposed into quantum gates.

The paper introduces a novel numerical method to mitigate errors in quantum circuits. The key algorithm of
this method involves finding the inverse of the Matrix Product Operator (MPO) in Step 2. The MPO is formed by
representing the super-operator associated with the quantum circuit. In this context, the MPO serves as a
compact representation of the super-operator, which describes the evolution of the quantum system under the
quantum circuit's operations. By finding the inverse of the MPO, the paper's proposed method aims to
efficiently correct errors and improve the overall performance and accuracy of the quantum circuit.

## Computer Implementation

See another folder named "Noisy quantum circuit simulator with MPDO".

## Physical Implementation

Indeed, calculating the inverse of a high-dimensional matrix is known to be an exponentially time-consuming
problem. However, with the advent of novel classical simulation techniques like tensor networks, the time
complexity can be significantly reduced to polynomial time using efficient contraction schemes. This
breakthrough allows for remarkable advancements in processing quantum calculations.

Tensor network methods provide a powerful framework for representing and manipulating large quantum states and
operators efficiently. By exploiting the underlying structure of quantum systems, tensor networks allow for
more compact and tractable representations of quantum states, reducing the computational overhead associated
with high-dimensional matrices. With tensor network techniques, quantum calculations that were once infeasible
due to their exponential time complexity can now be tackled efficiently, enabling simulations of more complex
and larger quantum systems. This has opened up exciting possibilities for exploring quantum phenomena and
solving quantum problems beyond the capabilities of classical computation.