import numpy as np 
from tqdm import tqdm
from .gates import Gates
from copy import copy

class QuantumComputer_psi:
    
    def __init__(self, N, gates, dtype=np.complex128):
        self.N = N
        self.dtype = dtype
        self.I = np.array([[1, 0],[0, 1]], dtype=self.dtype)
        self.gates = [] # [qubit, gate]
        if len(gates.shape) == 3:
            for i in range(N):
                gates_tmp = []
                for g in gates:
                    gates_tmp.append(self.make_gate(g, i))
                self.gates.append(gates_tmp)
        elif len(gates.shape) == 4:
            for i in range(N):
                gates_tmp = []
                for j in range(gates.shape[1]):
                    gates_tmp.append(self.make_gate(gates[i,j], i))
                self.gates.append(gates_tmp)
        else:
            print('wrong shape of the gates')
        self.gates = np.stack(self.gates)


    def make_gate(self, T, q):  
        if self.N == 1:
            return T
        elif self.N == 2:
            if q == 0:
                return(np.kron(T, self.I))
            elif q == 1:
                return(np.kron(self.I, T))
    
    
    def apply_gate_sequence(self, gate_sequence_ind, G, Gconst_flag, Gparams, n1Q, n2Q, nrepeat):
        '''
        gate_sequence_ind : array[random_ciruit_num, qubit_num, gate_num]
        '''
        if Gconst_flag:
            if gate_sequence_ind.shape[1] == 2:
                psi = np.array([[1], [0], [0], [0]], dtype=self.dtype)
                psi = psi[np.newaxis]
                for i in range(gate_sequence_ind.shape[2]): 
                    psi = G[:,np.newaxis]@((self.gates[0][gate_sequence_ind[:,0,i]])[np.newaxis]@
                                          ((self.gates[1][gate_sequence_ind[:,1,i]])[np.newaxis]@psi))
            elif gate_sequence_ind.shape[1] == 1:
                psi = np.array([[1], [0]], dtype=self.dtype)
                psi = psi[np.newaxis]
                for i in range(nrepeat):
                    psi = ((self.gates[0][gate_sequence_ind[:,0,i]])[np.newaxis]@psi)
                for i in range((gate_sequence_ind.shape[2]-nrepeat)//n1Q): 
                    for i1Q in range(n1Q):
                        psi = ((self.gates[0][gate_sequence_ind[:,0,nrepeat+i*n1Q+i1Q]])[np.newaxis]@psi)
                    for _ in range(n2Q):
                        psi = G[:,np.newaxis]@psi
                    # psi = G[:,np.newaxis]@((self.gates[0][gate_sequence_ind[:,0,i]])[np.newaxis]@psi)
        else:
            if gate_sequence_ind.shape[1] == 2:
                psi = np.array([[1], [0], [0], [0]], dtype=self.dtype)
                psi = psi[np.newaxis]
                for i in range(nrepeat):
                    psi = G@((self.gates[0][gate_sequence_ind[:,0,i]])[np.newaxis]@
                                          ((self.gates[1][gate_sequence_ind[:,1,i]])[np.newaxis]@psi))
                for i in range((gate_sequence_ind.shape[2]-nrepeat)//n1Q):
                    for i1Q in range(n1Q):
                        psi = ((self.gates[0][gate_sequence_ind[:,0,nrepeat+i*n1Q+i1Q]])[np.newaxis]@
                              ((self.gates[1][gate_sequence_ind[:,1,nrepeat+i*n1Q+i1Q]])[np.newaxis]@psi))
                    for _ in range(n2Q):
                        G = self.generate_G([*Gparams])
                        psi = G@psi
            elif gate_sequence_ind.shape[1] == 1:
                psi = np.array([[1], [0]], dtype=self.dtype)
                psi = psi[np.newaxis]
                for i in range(nrepeat):
                    psi = ((self.gates[0][gate_sequence_ind[:, 0, i]])[np.newaxis] @ psi)
                for i in range((gate_sequence_ind.shape[2] - nrepeat) // n1Q):
                    for i1Q in range(n1Q):
                        psi = ((self.gates[0][gate_sequence_ind[:, 0, nrepeat + i * n1Q + i1Q]])[np.newaxis] @ psi)
                    for _ in range(n2Q):
                        G = self.generate_G([*Gparams])
                        psi = G@psi
            else:
                print('wrong gate_sequence_ind.shape for Gconst_flag==False')


        return np.sum(np.asarray(np.abs(psi)**2), axis=-1)



    def generate_G(self, params):
        print('To use Gconst_flag=False you need to define generate_G function in xeb.QC')