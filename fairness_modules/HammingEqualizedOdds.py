import pandas as pd
import numpy as np
import gurobipy as gp
from .FairnessModule import FairnessModule

class HammingEqualizedOdds(FairnessModule):
    
    def __init__(self, args = {}):
        super().__init__()
        
        if 'group' not in args:
            raise Exception('No group assignments!')
        
        
        self.group = args['group']
        
        self.fairConstNames = ['posUbFair','posLbFair', 'negUbFair', 'negLbFair']

        self.eps = args['epsilon'] if 'epsilon' in args else 0.05
        return
        
    def defineObjective(self, delta, z, Y, args):
        '''
        Returns gurobi objective for pricing problem
        '''
        if 'lam' not in args or 'mu' not in args or 'fairDuals' not in args or 'row_samples' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Definition.')
        
        if 'posUbFair' not in args['fairDuals'] or 'posLbFair' not in args['fairDuals'] or \
            'negUbFair' not in args['fairDuals'] or 'negLbFair' not in args['fairDuals']:
            raise Exception('Required fairness dual variables not supplied for NoFair Objective Definition.')
        
        g = self.group[args['row_samples']]
        coeff_1 = 1 + (args['fairDuals']['negUbFair'] - args['fairDuals']['negLbFair'])/sum(g & ~Y)
        coeff_2 = 1 + (args['fairDuals']['negLbFair'] - args['fairDuals']['negUbFair'])/sum(~g & ~Y)

        objective = gp.LinExpr(np.ones(sum(~Y)), np.array(delta)[~Y]) #Y = False misclass term
        objective.add(gp.LinExpr(np.array(args['mu'])*-1, np.array(delta)[Y])) #Y = True misclass term
        objective.add(gp.LinExpr(args['lam']*np.ones(len(z)), z)) #Complexity term
        objective.add(coeff_1*sum(np.array(delta)[~Y & g]))
        objective.add(coeff_1*sum(np.array(delta)[~Y & ~g]))

        return objective
  
    
    def computeObjective(self, X, Y, features, args):
        '''
        Returns reduced cost for pricing problem for given inputs
        '''
        if 'lam' not in args or 'mu' not in args or 'fairDuals' not in args or 'row_samples' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Computation.')
        
        if 'ubFair' not in args['fairDuals'] or 'lbFair' not in args['fairDuals']:
            raise Exception('Required fairness dual variables not supplied for NoFair Objective Computation.')
        
        classPos = np.all(X[:,features],axis=1)
        g = self.group[args['row_samples']]
        coeff_1 = 1 + (args['fairDuals']['negUbFair'] - args['fairDuals']['negLbFair'])/sum(g & ~Y)
        coeff_2 = 1 + (args['fairDuals']['negLbFair'] - args['fairDuals']['negUbFair'])/sum(~g & ~Y)

        return args['lam']*(1+len(features)) - np.dot(classPos[Y],args['mu']) + sum(classPos[~Y]) + \
               coeff_1*sum(classPos[g & ~Y]) + \
               coeff_2*sum(classPos[~g & ~Y])

    
    
    def updateFairnessConstraint(self, column, constraints, args):
        column.addTerms(1/sum(self.group[~args['Y']])*sum(args['K_z'][self.group[~args['Y']]]) - \
                        1/sum(~self.group[~args['Y']])*sum(args['K_z'][~self.group[~args['Y']]]),
                        constraints['negUbFair'])
        column.addTerms(-1/sum(self.group[~args['Y']])*sum(args['K_z'][self.group[~args['Y']]]) + \
                        1/sum(~self.group[~args['Y']])*sum(args['K_z'][~self.group[~args['Y']]]),
                        constraints['negLbFair'])
        return

    
    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        constraints = {}
        x = np.array(x)
        g = self.group[Y]

        constraints['posUbFair'] = model.addConstr( 1/sum(g)*sum(x[g]) - 1/sum(~g)*sum(x[~g]) <= self.eps, 
                                                name = 'posUbFair')
        constraints['posLbFair'] = model.addConstr( -1/sum(g)*sum(x[g]) + 1/sum(~g)*sum(x[~g])<= self.eps, 
                                                name = 'posLbFair')
        constraints['negUbFair'] = model.addConstr( 0*x[0] <= self.eps, 
                                                name = 'negUbFair')
        constraints['negLbFair'] = model.addConstr( 0*x[0] <= self.eps, 
                                                name = 'negLbFair')

        return constraints
