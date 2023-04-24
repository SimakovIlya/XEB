class HDAWG_PRNG:
    '''
    PseudoRandom Number Generator used in HDAWG

    Methods
    -------
    next                            : return random int
    '''
    def __init__(self, seed=0xcafe, lower=0, upper=2**16-1):
        '''
        args
        ------
        seed = 0xcafe
        lower = 0
        upper = 2**16-1
        '''
        self.lsfr = seed
        self.lower = lower
        self.upper = upper
  



    def next(self):
        lsb = self.lsfr & 1
        self.lsfr = self.lsfr >> 1
        if (lsb):
            self.lsfr = 0xb400 ^ self.lsfr
        rand = ((self.lsfr * (self.upper-self.lower+1) >> 16) + self.lower) & 0xffff
        return rand