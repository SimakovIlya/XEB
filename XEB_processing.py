import numpy as np
from copy import copy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .simulator import QuantumComputer_psi
from .prng import HDAWG_PRNG
from tqdm import tqdm



class XEB_processing:
    '''
    Class for processing xeb data

    Methods
    -------
    get_measurement_for_XEB
    get_fidelity
    fit_depolarizing_fidelity
    reset_QC                        : change type of QC to 'psi' or 'ptm'

    get_random_gate_sequence_list   : return random gate_sequence_list for XEB
    get_prng_gate_sequence_list     : return prng gate_sequence_list for XEB
    check_gate_sequence_list        : plot frequency distribution histogram
    '''
    def __init__(self, N, gates, n1Q=1, n2Q=1, nrepeat=0):
        '''
        N           : number of qubits
        gates       : list of single-qubit gates
        '''
        self.N = N 
        self.QC = QuantumComputer_psi(N, gates)
        self.gates_psi = gates
        self.gates = gates
        self.n1Q = n1Q
        self.n2Q = n2Q
        self.nrepeat = nrepeat





    def get_measurement_for_XEB(self, G, gate_sequence_list, Gconst_flag=True, Gparams=None):
        '''
        args
        --------
        G                    : list of gates for xeb (or single gate)
        gate_sequence_list   : array of random gate indeces per each
                               [circuit depth, random circuit, qubit, gate indeces]
        progress             : show progress bar

        return diag_el [circuit_depth_index, n_random_circuit, measurements (00, 01, 10, 00)]
        '''
        G = np.asarray(G)
        if len(G.shape) != 3:
            G = G[np.newaxis]

        diag_el = []
        for gate_sequence_ind in gate_sequence_list:
            diag_el_per_circuit_depth = self.QC.apply_gate_sequence(gate_sequence_ind, G, Gconst_flag, Gparams, self.n1Q, self.n2Q, self.nrepeat)
            diag_el.append(diag_el_per_circuit_depth)
        if len(diag_el) >= 2:
            if diag_el[0].shape[1] != diag_el[1].shape[1]:
                diag_el[0] = np.tile(diag_el[0], (1,diag_el[1].shape[1],1))
        diag_el = np.asarray(diag_el)
        if diag_el.shape[1] == 1:
            diag_el = np.sum(diag_el, axis=1)

        return np.asarray(diag_el)




    def get_fidelity(self, diag_el, diag_el_exp): 
        assert len(diag_el_exp.shape) == 3
        if len(diag_el.shape) == 3:
            diag_el = diag_el[:,np.newaxis]
        e_u = np.sum(diag_el_exp**2, axis = 2)[:,np.newaxis]
        u_u = np.sum(diag_el_exp, axis = 2)[:,np.newaxis] / 2**self.N
        m_u = np.sum(diag_el_exp[:,np.newaxis] * diag_el, axis = -1)
        
        depol_fidelity_lsq = (np.sum((m_u-u_u)*(e_u-u_u),axis=-1) / np.sum((e_u-u_u)**2, axis=-1)).T
        # fidelity = depol_fidelity_lsq + (1-depol_fidelity_lsq)/diag_el.shape[-1]#2**self.N
        return depol_fidelity_lsq
    
    
    
    
    def fit_depolarizing_fidelity(self, circuit_depths, fidelity):
        
        def dep_fidelity_curve(circuit_depths, pc, b):
            return np.power(pc, circuit_depths) + b
        
        popt, pcov = curve_fit(dep_fidelity_curve, circuit_depths, fidelity)
        return popt, pcov




    def get_random_gate_sequence_list(self, n_random_circuits, circuit_depths):
        '''
        args
        --------
        n_random_circuits   : number of random circuits (int)
        circuit_depths      : ndarray (or list)

        return gate_sequence_list [circuit depth, random circuit, qubit, gate indeces] (list)
        '''
        gate_sequence_list = []
        for circuit_depth in circuit_depths:
            gate_sequence_list.append(np.random.randint(len(self.gates), 
                                                        size=(n_random_circuits, self.N, circuit_depth)))
        return gate_sequence_list




    def get_prng_gate_sequence_list(self, seeds, circuit_depths):
        '''
        args
        --------
        seeds               : array of random seeds per each
                              [circuit depth, random circuit, qubit]
        circuit_depths      : ndarray (or list)

        return gate_sequence_list [circuit depth, random circuit, qubit, gate indeces] (list)
        '''
        assert self.N == seeds.shape[2]
        gate_sequence_list = []
        n_random_circuits = seeds.shape[1]
        for j, circuit_depth in enumerate(circuit_depths):
            gate_sequence = np.zeros((n_random_circuits, self.N, circuit_depth), dtype=int)
            for i in range(n_random_circuits):
                for qubit in range(self.N):
                    prng = HDAWG_PRNG(seed=seeds[j, i, qubit], lower=0, upper=self.gates.shape[-3]-1)
                    for k in range(circuit_depth):
                        gate_sequence[i, qubit, k] = prng.next()
                    # print(seeds[j, i, qubit], gate_sequence[i, qubit], len(self.gates)-1)
            gate_sequence_list.append(gate_sequence)
        return gate_sequence_list




    def check_gate_sequence_list(self, gate_sequence_list):
        '''
        args
        ------
        gate_sequence_list   : array of random gate indeces per each
                               [circuit depth, random circuit, qubit, gate indeces]

        return frequency distribution histogram
        '''
        b = []
        for gate_sequence in gate_sequence_list:
            b.append(np.bincount(gate_sequence.ravel(), minlength=len(self.gates)))
        b = np.sum(np.asarray(b), axis=0)
        plt.hist(np.arange(len(self.gates)), bins=len(self.gates), weights=b, alpha=0.7)
        plt.title('check for uniform distribution')
        plt.xlabel('gate number')
        plt.ylabel('gate applied')
        plt.show()