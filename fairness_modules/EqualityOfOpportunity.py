import pandas as pd
import numpy as np
import gurobipy as gp
from .FairnessModule import FairnessModule

class EqualityOfOpportunity(FairnessModule):
    
    def __init__(self, args = {}):
        super().__init__()
        
        if 'group' not in args:
            raise Exception('No group assignments!')
        
        
        self.group = args['group']
        
        self.fairConstNames = ['ubFair','lbFair']

        self.eps = args['epsilon'] if 'epsilon' in args else 0.05
        return
        
    def defineObjective(self, delta, z, Y, args):
        '''
        Returns gurobi objective for pricing problem
        '''
        if 'lam' not in args or 'mu' not in args or 'fairDuals' not in args or 'row_samples' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Definition.')
        
        if 'ubFair' not in args['fairDuals'] or 'lbFair' not in args['fairDuals']:
            raise Exception('Required fairness dual variables not supplied for NoFair Objective Definition.')
        
        g = self.group[args['row_samples']]
        coeff_g_1 = (args['fairDuals']['ubFair'] - args['fairDuals']['lbFair'])*1/sum(g & Y)
        coeff_g_2 = (args['fairDuals']['lbFair'] - args['fairDuals']['ubFair'])*1/sum(~g & Y)

        objective = gp.LinExpr(np.ones(sum(~Y)), np.array(delta)[~Y]) #Y = False misclass term
        objective.add(gp.LinExpr(np.array(args['mu'])*-1, np.array(delta)[Y])) #Y = True misclass term
        objective.add(gp.LinExpr(args['lam']*np.ones(len(z)), z)) #Complexity term
        objective.add(gp.LinExpr(coeff_g_1*np.ones(len(np.array(delta)[g & Y])), np.array(delta)[g & Y])) #1st fair term
        objective.add(gp.LinExpr(coeff_g_2*np.ones(len(np.array(delta)[~g & Y])), np.array(delta)[~g & Y])) #2nd fair term

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
        return args['lam']*(1+len(features)) - np.dot(classPos[Y],args['mu']) + sum(classPos[~Y]) + \
               (args['fairDuals']['ubFair'] - args['fairDuals']['lbFair'])*(1/sum(g & Y) *sum(classPos[g & Y])) + \
               (args['fairDuals']['lbFair'] - args['fairDuals']['ubFair'])*(1/sum(~g & Y) *sum(classPos[~g & Y])) 

    

    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        constraints = []
        x = np.array(x)
        g = self.group[Y]

        constraints.append(model.addConstr( 1/sum(g)*sum(x[g]) - \
                                                1/sum(~g)*sum(x[~g])<= self.eps, 
                                                name = 'ubFair'))
        constraints.append(model.addConstr( -1/sum(g)*sum(x[g]) + \
                                                1/sum(~g)*sum(x[~g])<= self.eps, 
                                                name = 'lbFair'))

        return constraints
