import numpy as np 

class Gates:
    '''
    List of Quantum Gates

    Methods
    -------
    I, X, Y, Z, S, H
    Rx, Ry, Rz
    fSim, SWAP, iSWAP, CNOT
    get_clifford_gates          : return list of 24 clifford gates
    '''
    def __init__(self, dtype=np.complex128):
        self.dtype = dtype
        
    def I(self):
        return np.array([[1, 0],[0, 1]], dtype=self.dtype)
    
    
    def X(self):
        return np.array([[0, 1], [1, 0]], dtype=self.dtype)
    
    
    def Y(self):
        return np.array([[0, -1j],[1j, 0]], dtype=self.dtype)
    
    
    def Z(self):
        return np.array([[1, 0],[0, -1]], dtype=self.dtype)

    
    def S(self, sign=1):
        return np.array([[1, 0], [0, sign*1j]], dtype=self.dtype)
    
    
    def H(self):
        return 1/np.sqrt(2)*np.array([[1, 1],[1, -1]], dtype=self.dtype)
    
    
    def Rx(self, theta):
        G = np.array([[np.cos(theta/2),    -1j*np.sin(theta/2)],
                      [-1j*np.sin(theta/2),  np.cos(theta/2)]], dtype=self.dtype)
        return G

    
    def Ry(self, theta):
        G = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                      [np.sin(theta/2),  np.cos(theta/2)]], dtype=self.dtype)
        return G  

    
    def Rz(self, theta):
        G = np.array([[1, 0],
                      [0, np.exp(1j*theta)]], dtype=self.dtype)
        return G


    def fSim(self, theta, phi):
        G = np.array([[1, 0, 0, 0],
                      [0, np.cos(theta), 1j*np.sin(theta), 0],
                      [0, 1j*np.sin(theta), np.cos(theta), 0],
                      [0, 0, 0, np.exp(-1j*phi)]], dtype=self.dtype)
        return G
    
    
    def SWAP(self):
        G = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=self.dtype)
        return G


    def CZ(self):
        G = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]], dtype=self.dtype)
        return G
    
    
    def CNOT(self, ctrl=0):
        if ctrl == 0:
            G = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]], dtype=self.dtype)
        elif ctrl == 1:
            G = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0]], dtype=self.dtype)
        return G
    
    
    def iSWAP(self):
        return self.fSim(-np.pi/2, 0)

    def get_clifford_gates(self):
        clif_gates = [
            self.I(), 
            self.X(),
            self.Y(), 
            self.Z(),

            self.Rx(theta = np.pi/2),
            self.Rx(theta = -np.pi/2),
            self.Ry(theta = np.pi/2),
            self.Ry(theta = -np.pi/2),
            self.Rz(theta = np.pi/2),
            self.Rz(theta = -np.pi/2),

            self.X()@self.Rz(theta = np.pi/2),
            self.Rx(theta = np.pi/2)@self.Z(),
            self.Rz(theta = np.pi/2)@self.X(),
            self.Rz(theta = np.pi/2)@self.Rx(theta = np.pi/2)@self.Rz(theta = np.pi/2),
            self.Z()@self.Rx(theta = np.pi/2),
            self.Rz(theta = 3*np.pi/2)@self.Rx(theta = np.pi/2)@self.Rz(theta = 3*np.pi/2),

            self.Rx(theta = np.pi/2)@self.Rz(theta = np.pi/2),
            self.Rz(theta = np.pi/2)@self.Rx(theta = np.pi/2),
            self.Rx(theta = np.pi/2)@self.Rz(theta = 3*np.pi/2),
            self.Rz(theta = np.pi/2)@self.Rx(theta = np.pi/2)@self.Rz(theta = 2*np.pi/2),
            self.Rz(theta = 2*np.pi/2)@self.Rx(theta = np.pi/2)@self.Rz(theta = np.pi/2),
            self.Rz(theta = 3*np.pi/2)@self.Rx(theta = np.pi/2),
            self.Rz(theta = 2*np.pi/2)@self.Rx(theta = np.pi/2)@self.Rz(theta = 3*np.pi/2),
            self.Rz(theta = 3*np.pi/2)@self.Rx(theta = np.pi/2)@self.Rz(theta = 2*np.pi/2)
        ]
        return np.asarray(clif_gates)







