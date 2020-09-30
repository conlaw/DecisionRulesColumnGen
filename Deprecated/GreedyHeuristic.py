import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from RuleGenerator import RuleGenerator

class GreedyHeuristic(RuleGenerator):
    '''
    Implementation of IP Pricing Problem Solver
    '''
    
    def __init__(self, ruleMod, args = {}):
        self.ruleMod = ruleMod
        
        #Set rule complexity if supplied in arguments
        self.ruleComplex = args['ruleComplexity'] if 'ruleComplexity' in args else 5
        self.numRulesToKeep = args['numRulesToKeep'] if 'ruleComplexity' in args else 20
        
        
    def generateRule(self, args):
        '''
        Solve the IP Pricing problem to generate new rule(s)
        '''
        
        #Retrieve parameters
        if 'lam' not in args or 'mu' not in args:
            raise Exception('Required arguments not supplied for DNF IP Rule Generator.')

        lam = args['lam']
        mu = args['mu']
                   
        verbose = args['verbose'] if 'verbose' in args else False
        
        feature_set = [[]]
        good_rules = []
        good_rule_obj = []

        for i in range(self.ruleComplex):
            newFeatures = []
            res = []
            for f in feature_set:
                for i in range(self.ruleMod.X.shape[1]):
                    #If the feature is already in the feature set move on
                    if i in f:
                        continue
                    
                    #Create new feature set
                    newF = np.concatenate([f, [i]]).astype(np.int64)
                    newFeatures.append(newF)
                    
                    #Compute objective
                    obj = self.computeObj(newF, lam, mu)
                    res.append(obj)
                    
                    #If reduced cost is negative add to rules
                    if obj < 0:
                        rule = np.zeros(self.ruleMod.X.shape[1])
                        rule[newF] = 1
                        good_rules.append(rule)
                        good_rule_obj.append(obj)
            
            #Adjust feature set to features of size i, best numKeep number
            feature_set = np.array(newFeatures)[np.argsort(res)][:self.numRulesToKeep] 
            
        #Only return rules with negative reduced costs
        if len(good_rules) > 0:        
            return True, np.array(good_rules)[np.argsort(good_rule_obj)][:self.numRulesToKeep]
        else:
            return False, []
            
    def computeObj(self, features, lam, mu):
        classPos = np.all(self.ruleMod.X[:,features],axis=1)
        return lam*(1+len(features)) - np.dot(classPos[self.ruleMod.Y],mu) + sum(classPos[~self.ruleMod.Y])


        
                    
        
            