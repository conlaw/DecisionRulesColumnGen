import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from RuleGenerator import RuleGenerator

class DNF_IP_RuleGenerator(RuleGenerator):
    '''
    Implementation of IP Pricing Problem Solver
    '''
    
    def __init__(self, ruleMod, args = {}):
        self.ruleMod = ruleMod
        
        #Set rule complexity if supplied in arguments
        self.ruleComplex = args['ruleComplexity'] if 'ruleComplexity' in args else 100
        
        self.model = gp.Model('masterLP')
        self.initModel()

    def initModel(self):
        '''
        Initialize model elements that don't vary with each iteration
        '''
        
        numSamples, numFeatures = self.ruleMod.X.shape
        
        #Construct decision variables for new rule 
        self.z = []
        for k in range(numFeatures):
            self.z.append(self.model.addVar(vtype=GRB.BINARY, name="z[%d]"%k))

        #Construct decision variables for misclassification
        self.delta = []
        for i in range(numSamples):
            self.delta.append(self.model.addVar(vtype=GRB.BINARY, name= "delta[%d]"%i))
        
        #Add complexity constraint
        self.complexConst = self.model.addConstr(gp.LinExpr(np.ones(numFeatures), self.z) <= self.ruleComplex,
                                                 name="ComplexityConst")
        #Add misclassification constraints
        for i in range(numSamples):
            #Constraint for data samples where Y = False
            if not self.ruleMod.Y[i]:
                 self.model.addConstr(gp.LinExpr(~self.ruleMod.X[i,:]*1, self.z) + self.delta[i] >= 1,
                                      name="sampleConstraint[%d]"%i)
            #Constraints for data samples where Y = True
            else:
                for j in range(numFeatures):
                    if not self.ruleMod.X[i,j]:
                        self.model.addConstr(self.delta[i]+self.z[j] <= 1,
                                             name="sampleConstraint[%d]-%d"%(i,j))

        
    def generateRule(self, args):
        '''
        Solve the IP Pricing problem to generate new rule(s)
        '''
        
        #Retrieve parameters
        if 'lam' not in args or 'mu' not in args:
            raise Exception('Required arguments not supplied for DNF IP Rule Generator.')

        lam = args['lam']
        mu = args['mu']
                   
        returnAllSolutions = args['returnAllSolutions'] if 'returnAllSolutions' in args else True
        verbose = args['verbose'] if 'verbose' in args else False
        
        #Create objective function
        objective = gp.LinExpr(np.ones(sum(~self.ruleMod.Y)), np.array(self.delta)[~self.ruleMod.Y]) #Y = False misclass term
        objective.add(gp.LinExpr(np.array(mu)*-1, np.array(self.delta)[self.ruleMod.Y])) #Y = True misclass term
        objective.add(gp.LinExpr(lam*np.ones(len(self.z)), self.z)) #Complexity term
        
        #Set the objective and determine output level
        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.Params.OutputFlag = verbose
        
        if 'timeLimit' in args:
            self.model.Params.TimeLimit = args['timeLimit']
        
        #Solve
        self.model.update()
        self.model.optimize()
        
        #Only return rules with negative reduced costs
        if self.model.objVal < 0:
            if verbose:
                for v in self.model.getVars():
                    print('%s %g' % (v.varName, v.x))
        
            return True, self.getAllRules() if returnAllSolutions else self.getBestRule()
        else:
            return False, []
            
    def getBestRule(self):
        '''
        Returns optimal solution
        '''
        return [[v.x for v in self.model.getVars()[0:len(self.z)]]]
    
    def getAllRules(self):
        '''
        Return all rules with negative reduced costs
        '''
        
        solCount = self.model.SolCount
        rules = []
        
        #Loop through stored solutions and keep if negative reduced cost
        for i in range(solCount):
            self.model.Params.SolutionNumber = i
            
            if self.model.getAttr(GRB.Attr.PoolObjVal) >= 0:
                break
            
            rules.append(self.model.getAttr(GRB.Attr.Xn)[0:len(self.z)])
        
        return rules




        
                    
        
            