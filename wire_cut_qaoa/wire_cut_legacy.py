# useful additional packages

from qiskit_optimization.applications import Maxcut
import warnings
warnings.filterwarnings("ignore")

# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit.circuit.library import QAOAAnsatz

from qiskit import transpile
from qiskit_aer import AerSimulator


from circuit_knitting.cutting.cutqc import cut_circuit_wires

from qiskit_ibm_runtime import Options
from circuit_knitting.cutting.cutqc import (
    reconstruct_full_distribution,
)
from circuit_knitting.cutting.cutqc import evaluate_subcircuits
from qiskit.result import ProbDistribution
from qiskit.primitives import BackendSampler




def maxcut_obj(solution, graph):
    """Given a bit string as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    Args:
        solution: (str) solution bit string
        graph: networkx graph
    Returns:
        obj: (float) Objective
    """
    # pylint: disable=invalid-name
    obj = 0
    for i, j in graph.edges():
        if solution[i] != solution[j]:
            obj -= 1
    return obj



def compute_expectation(counts, graph):
    """Computes expectation value based on measurement results
    Args:
        counts: (dict) key as bit string, val as count
        graph: networkx graph
    Returns:
        avg: float
             expectation value
    """
    avg = 0
    sum_count = 0
    for bit_string, count in counts.items():
        obj = maxcut_obj(bit_string, graph)
        avg += obj * count
        sum_count += count
    return avg/sum_count



def get_expectation(graph,reps, width, num_cuts, num_subcircuits):
    """Get the expectation value of the graph
    Args:
        graph: networkx graph
        reps: (int) number of repetitions
        width: (int) width of the subcircuits
        num_cuts: (int) number of cuts
        num_subcircuits: (list[int]) number of subcircuits
    Returns:
        float: expectation value
    """
    def execute_circ(params):
        max_cut = Maxcut(graph)
        qp = max_cut.to_quadratic_program()
        qubitOp, offset = qp.to_ising()
        # QAOA ansatz circuit
        ansatz = QAOAAnsatz(qubitOp, reps=reps)
        circuit_basis = transpile(ansatz, basis_gates=['u3', 'cx'], optimization_level=3)
        num_qubits = circuit_basis.num_qubits
        circuit_basis = circuit_basis.assign_parameters(params)
        cuts = cut_circuit_wires(
            circuit=circuit_basis,
            method="automatic",
            max_subcircuit_width=width,
            max_cuts=num_cuts,
            num_subcircuits=num_subcircuits,
        )

        options = Options(execution={"shots": 1000})

        # Run 2 parallel qasm simulator threads
        backend_names = ["statevector_simulator","ibmq_qasm_simulator"] 
        #backend_names = ["ibmq_qasm_simulator"] * len(cuts["subcircuits"])

        #subcircuit_instance_probabilities = evaluate_subcircuits(cuts)
        subcircuit_instance_probabilities = evaluate_subcircuits(cuts,
                                                         backend_names=backend_names,
                                                         options=options,
                                                        )
        
        reconstructed_probabilities = reconstruct_full_distribution(
            circuit_basis, subcircuit_instance_probabilities, cuts)
        reconstructed_distribution = {
        i: prob for i, prob in enumerate(reconstructed_probabilities)}

        reconstructed_dict_bitstring = ProbDistribution(
            data=reconstructed_distribution).binary_probabilities(num_bits=num_qubits)
        
        return compute_expectation(reconstructed_dict_bitstring, graph)
    


    return execute_circ
    

    

def reconstruct_result(graph,reps,opt_params):
    """
    Args:
        graph: networkx graph
        reps: (int) number of repetitions
        opt_params: (list[float]) optimized parameters
    """

    max_cut = Maxcut(graph)
    qp = max_cut.to_quadratic_program()
    qubitOp, offset = qp.to_ising()
    # QAOA ansatz circuit
    ansatz = QAOAAnsatz(qubitOp, reps=reps)
    circuit = ansatz.assign_parameters(opt_params)
    sampler = BackendSampler(backend=AerSimulator(method="statevector"))
    circuit.measure_all()
    samp_dist = sampler.run(circuit).result().quasi_dists[0]
    my_dict = dict(sorted(samp_dist.binary_probabilities().items(),key=lambda item: item[1], reverse=True))

    return my_dict




