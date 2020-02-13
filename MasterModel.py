import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class MasterModel(object):
    '''
    Object to contain and run the restricted model
    '''
    
    def __init__(self, rule_mod, complexity = 40):
        #Set-up constants
        self.ruleModel = rule_mod
        self.complexityConstraint = complexity
        self.w = {}
        self.var_counter = 0
                
        #Initialize Model
        self.model = gp.Model('masterLP')
        self.initModel()
    
    def initModel(self):
        '''
        Function to initialize the base restricted model with no rules
        '''
        
        #Initialize positive misclassification variables
        self.x = {}
        for k in range(sum(self.ruleModel.Y)):
            self.x[k] = self.model.addVar(obj=1, vtype=GRB.BINARY, name="eps[%d]"%k)

        #Add positive misclassification constraints
        self.misClassConst = []
        for i in range(len(self.x)):
            self.misClassConst.append(self.model.addConstr(self.x[i] >= 1, name="MisclassConst[%d]"%i))
 
        #Add complexity constraint
        self.compConst = self.model.addConstr( 0*self.x[0] <= self.complexityConstraint, name = 'compConst')

    
    def solve(self, relax = True, verbose = False):
        '''
        Function to solve the restricted model.
        - Solves the relaxed LP if relax = True
        - Returns the final optimized model object
        '''
        
        #Update model, select version to run and optimize
        self.model.update()
        finalMod = self.model.relax() if relax else self.model #Can put something else instead of base MIP solver
        finalMod.Params.OutputFlag = verbose
        finalMod.optimize()
        
        #Print results if verbose
        if verbose:
            for v in finalMod.getVars():
                print('%s %g' % (v.varName, v.x))
            
            print('Obj: %g' % finalMod.objVal)

        return finalMod.optimize()
        
    def addRule(self, rules): 
        '''
        Function to add new rules to the restricted model.
        -Input takes LIST of rule objects
        '''
        
        #Need to deal with case when added rule not unique
        K_p, K_z_coeff, c = self.ruleModel.addRule(rules)
        
        #Add new decision variable for each rule
        for i in range(len(c)):
            
            #Specify new column
            newCol = gp.Column()
            newCol.addTerms(K_p[:,i] , self.misClassConst)
            newCol.addTerms(c[i],  self.compConst)
            
            #Add decision variable
            self.w[self.var_counter] = self.model.addVar(obj=K_z_coeff[i], 
                                           vtype=GRB.BINARY, 
                                           name="w[%d]"%self.var_counter, 
                                           column=newCol)
            self.var_counter += 1


        
        
        