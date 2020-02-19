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
        self.model_count = 0
                
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

    
    def solve(self, relax = True, verbose = False, saveModel = False):
        '''
        Function to solve the restricted model.
        - Solves the relaxed LP if relax = True
        - Returns the final optimized model object
        '''
                
        #Update model, select version to run and optimize
        self.model.update()
        self.finalMod = self.model.relax() if relax else self.model #Can put something else instead of base MIP solver
        self.finalMod.Params.OutputFlag = verbose
        self.finalMod.optimize()
        self.model_count += 1
        
        if saveModel:
            self.finalMod.write('model-'+str(self.model_count)+'.lp')
        
        #Print results if verbose
        if verbose:
            for v in self.finalMod.getVars():
                print('%s %g' % (v.varName, v.x))
            
            print('Obj: %g' % self.finalMod.objVal)

        #Construct results dictionary
        results = {}
        results['model'] = self.finalMod
        results['ruleSet'] = self.getRuleSet(self.finalMod.getVars())

        if relax:
            mu = []
            lam = None
            
            #Recover Dual Variables if using LP Relaxation
            for c in self.finalMod.getConstrs():
                if c.ConstrName == 'compConst':
                    lam = c.Pi
                else:
                    mu.append(c.Pi)
            
            results['mu'] = mu
            results['lam'] = lam

        
        return results
        
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

    
    def getRuleSet(self, decisionVars):
        '''
        Given final decision variables, returns the optimal rules as determined by the model
        '''
        
        # For rules generated during relaxed version, incldues all rules where w > 0
        inclRules = [v.x > 0 for v in decisionVars[len(self.x)::]]
        
        # Return what we can given the current state of variables
        if len(inclRules) > 0 and self.ruleModel.rules is not None: 
            return self.ruleModel.rules[inclRules]
        else:
            return []
            
        
        
        

        
        
        