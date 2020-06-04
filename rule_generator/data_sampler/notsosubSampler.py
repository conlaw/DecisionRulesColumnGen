from .SubSampler import SubSampler
import numpy as np

class NotSoSubSampler(SubSampler):
    
    def getSample(self, X, Y, mu, alpha, args = {}):
        '''
        Takes a set of rules and returns K_p, and K_z coefficient
        - Needs to be specified in the child class
        '''
        return X, Y, mu, alpha, np.ones(X.shape[0]).astype(np.bool),  np.ones(X.shape[1]).astype(np.bool)      
