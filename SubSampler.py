class SubSampler(object):
    '''
    Parent class to object that generates new rules.
    To add a new type of RuleModel:
        - Create a new child class
        - Specify how to generate a rule (taking in various arguments)
    '''
    
    def __init__(self, args = {}):
        pass
        
    def getSample(self, X, Y, mu, args = {}):
        '''
        Takes a set of rules and returns K_p, and K_z coefficient
        - Needs to be specified in the child class
        '''
        pass            

        
            