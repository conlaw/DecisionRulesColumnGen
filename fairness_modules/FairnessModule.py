class FairnessModule(object):
    
    def __init__(self, args = {}):
        self.fairConstNames = []
        self.fairDuals = {}
        pass
        
    def defineObjective(self, rules, reduced_costs, col_samples):
        '''
        Returns gurobi objective for pricing problem
        '''
        pass  
    
    def computeObjective(self, rules, reduced_costs, col_samples):
        '''
        Returns reduced cost for pricing problem for given inputs
        '''
        pass  
    
    def extractDualVariables(self, constraint):
        '''
        Returns dict with dual variables related to fairness constraint
        '''
        self.fairDuals[constraint.ConstrName] = constraint.Pi
        
        return

    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        pass
    
    def updateFairnessConstraint(self, column, constraints, args):
        return
        
            

