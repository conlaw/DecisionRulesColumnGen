from SubSampler import SubSampler

class NotSoSubSampler(SubSampler):
    
    def getSample(self, X, Y, mu, args = {}):
        '''
        Takes a set of rules and returns K_p, and K_z coefficient
        - Needs to be specified in the child class
        '''
        return X, Y, mu          
