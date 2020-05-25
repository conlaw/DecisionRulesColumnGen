import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

class MasterModel(object):
    '''
    Object to contain and run the restricted model
    '''
    
    def __init__(self, rule_mod, fairnessModule, args = {}):
        #Set-up constants
        self.ruleModel = rule_mod
        self.fairnessModule = fairnessModule
        self.complexityConstraint = args['ruleComplexity'] if 'ruleComplexity' in args else 40
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
        self.x = []
        for k in range(sum(self.ruleModel.Y)):
            self.x.append(self.model.addVar(obj=1, vtype=GRB.BINARY, name="eps[%d]"%k))

        #Add positive misclassification constraints
        self.misClassConst = []
        for i in range(len(self.x)):
            self.misClassConst.append(self.model.addConstr(self.x[i] >= 1, name="MisclassConst[%d]"%i))
 
        #Add complexity constraint
        self.compConst = self.model.addConstr( 0*self.x[0] <= self.complexityConstraint, name = 'compConst')
        
        self.fairnessConstraints = self.fairnessModule.createFairnessConstraint(self.model, self.x, self.ruleModel.Y)
    
    def solve(self, relax = True, verbose = False, saveModel = False):
        '''
        Function to solve the restricted model.
        - Solves the relaxed LP if relax = True
        - Returns the final optimized model object
        '''
                
        #Update model, select version to run and optimize
        self.model.update()
        self.finalMod = self.model.relax() if relax else self.model #Can put something else instead of base MIP solver
        
        #Need to un-hard code this later
        if not relax:
            self.finalMod.Params.TimeLimit = 300
            
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
        results['obj'] = self.finalMod.objVal
        results['ruleSet'] = self.getRuleSet(self.finalMod.getVars())

        if relax:
            mu = []
            lam = None
            
            #Recover Dual Variables if using LP Relaxation
            for c in self.finalMod.getConstrs():
                if c.ConstrName == 'compConst':
                    lam = c.Pi
                elif c.ConstrName in self.fairnessModule.fairConstNames:
                    self.fairnessModule.extractDualVariables(c)
                else:
                    mu.append(c.Pi)
            
            results['mu'] = mu
            results['lam'] = lam
            results['fairDuals'] = self.fairnessModule.fairDuals

        
        return results
    
    def getRC(self):
        decisionVars = self.finalMod.getVars()
        return [v.getAttr("RC") for v in decisionVars if 'w' in v.getAttr("VarName")]
    
    def resetModel(self, initialRules = None):
        self.model = gp.Model('masterLP')
        print('init model')
        self.initModel()
        print('adding rules')
        self.addRule(initialRules)
        print('done')

        return
        
        
    def addRule(self, rules): 
        '''
        Function to add new rules to the restricted model.
        -Input takes LIST of rule objects
        '''
        
        start_time = time.perf_counter()

        #Need to deal with case when added rule not unique
        K_p, K_z_coeff, c, K_z= self.ruleModel.addRule(rules)
        
        print('Rule model adding rules took %.2f seconds'%(time.perf_counter() - start_time))
        start_time = time.perf_counter()

        #Add new decision variable for each rule
        for i in range(len(c)):
            
            #Specify new column
            newCol = gp.Column()
            newCol.addTerms(K_p[:,i] , self.misClassConst)
            newCol.addTerms(c[i],  self.compConst)
            self.fairnessModule.updateFairnessConstraint(newCol,self.fairnessConstraints,{'K_z':K_z[:,i],
                                                                                         'Y': self.ruleModel.Y})
            
            #Add decision variable
            self.w[self.var_counter] = self.model.addVar(obj=K_z_coeff[i], 
                                           vtype=GRB.BINARY, 
                                           name="w[%d]"%self.var_counter, 
                                           column=newCol)
            self.var_counter += 1
        print('Adding columns took %.2f seconds'%(time.perf_counter() - start_time))

    
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
        
    def getAllSolutions(self):
        '''
        Return rule with best accuracy
        '''
        
        solCount = self.model.SolCount
        solutions = []
        objs = []
        
        #Loop through stored solutions and keep if negative reduced cost
        for i in range(solCount):
            self.model.Params.SolutionNumber = i
            solutions.append(self.getRuleSetNumpy(self.finalMod.getAttr(GRB.Attr.Xn)))
        
        print('Number of solutions returned: ', len(solutions))
        return solutions

    def getRuleSetNumpy(self, decisionVars):
        '''
        Given final decision variables, returns the optimal rules as determined by the model
        '''
        
        # For rules generated during relaxed version, incldues all rules where w > 0
        inclRules = [v > 0 for v in decisionVars[len(self.x)::]]
        
        # Return what we can given the current state of variables
        if len(inclRules) > 0 and self.ruleModel.rules is not None: 
            return self.ruleModel.rules[inclRules]
        else:
            return []
        
        
        

        
        
        