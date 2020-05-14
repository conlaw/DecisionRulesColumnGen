import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from DNFRuleModel import DNFRuleModel
from MasterModel import MasterModel
from rule_generator.GeneralRuleGenerator import GeneralRuleGenerator
from fairness_modules import *

class Classifier(object):
    '''
    Object to create a binary classifier using column generation framework
    '''
    
    def __init__(self, X, Y,
                 args = {},
                 ruleModel = 'DNF',
                 ruleGenerator = 'Generic',
                 fairness_module = 'unfair'):
        
        #Define class variables
        self.ruleMod = None
        self.ruleGen = None
        self.master = None
        self.fitRuleSet = None
        self.numIter = 0
        self.args = args
        self.mip_results = []
        self.final_mip = 0
        self.final_ip = 0
        fairness_module = args['fairness_module'] if 'fairness_module' in args else fairness_module
        
        # Map parameters to instantiated objects
        self.initFairnessModule(fairness_module)
        self.initRuleModel(X, Y, ruleModel)
        self.initRuleGenerator(ruleGenerator)
        self.master = MasterModel(self.ruleMod, self.fairnessModule, self.args)
        
    def fit(self, initial_rules = None, verbose = False, timeLimit = None, timeLimitPricing = None, colGen = True):
        '''
        Function to generate a rule set
            - Can take initial set of rules
            - Verbose parameter controls how much output is displayed during intermediary steps
        '''
        
        # Add initial rules to master model
        if initial_rules is not None:
            self.master.addRule(initial_rules)
        
        if timeLimit is not None:
            start_time = time.perf_counter()
        
        while colGen:
            self.numIter += 1
            # Solve relaxed version of restricted problem
            if verbose:
                print('Solving Restricted LP')
            results = self.master.solve(verbose = verbose, relax = True)
            results['verbose'] = verbose
            self.mip_results.append(results['obj'])
            
            if timeLimitPricing is not None:
                results['timeLimit'] = timeLimitPricing
            
            # Generate new candidate rules
            if verbose:
                print('Generating Rule')
            ruleFlag, rules = self.ruleGen.generateRule(results)
            
            # If no new rules generated exit out and solve master to optimality
            if ruleFlag:
                if verbose:
                    print('Adding %d new rule(s)'%len(rules))
                self.master.addRule(rules)
            else:
                if verbose:
                    print('No new rules generated.')
                self.master.addRule(np.array([np.ones(self.ruleMod.X.shape[1])]))
            
            if timeLimit is not None: 
                if time.perf_counter() - start_time > timeLimit:
                    print('Time limit for column generation exceeded. Solving MIP.')
                    break
        
        # Solve master problem to optimality
        if verbose:
            print('Solving final master problem to integer optimality')
        
        results = self.master.solve(verbose = verbose, relax = False)
                    
        self.fitRuleSet = results['ruleSet']
        self.final_mip = self.mip_results[-1] if colGen else -1
        self.final_ip = results['obj']
        
        #Return final rules
        return self
        
    def predict(self, X):
        '''
        Function to predict class labels using the fitted rule set
        '''
        if self.fitRuleSet is None:
            raise Exception("Model not fit. Can't make inference!")
        
        if len(self.fitRuleSet) == 0:
            return np.repeat(sum(self.ruleMod.Y) >= len(self.ruleMod.Y)/2, X.shape[0])
        
        return self.ruleMod.predict(X, self.fitRuleSet)
        
    def initRuleModel(self, X, Y, ruleModel):
        '''
        Function that maps string rule models to objects
           - To add a new rule model simply add the object to the if control flow
        '''
        
        if ruleModel == 'DNF':
            self.ruleMod = DNFRuleModel(X, Y)
        else:
            raise Exception('No associated rule model found.')
    
    def initRuleGenerator(self, ruleGenerator):
        '''
        Function that maps string rule generators to objects
           - To add a new rule generator simply add the object to the if control flow
        '''

        if ruleGenerator == 'Generic':
            self.ruleGen = GeneralRuleGenerator(self.ruleMod, self.fairnessModule, self.args)
        else:
            raise Exception('No associated rule generator found.')
            
    def initFairnessModule(self, fairnessModule):
        '''
        Function that maps string fairness modules to objects
           - To add a new fairness module simply add the object to the if control flow
        '''
        print(fairnessModule)
        if fairnessModule == 'unfair':
            self.fairnessModule = NoFair.NoFair(self.args)
        elif fairnessModule == 'EqOfOp':
            self.fairnessModule = EqualityOfOpportunity.EqualityOfOpportunity(self.args)
        elif fairnessModule == 'HammingDisp':
            self.fairnessModule = HammingDisparity.HammingDisparity(self.args)
        elif fairnessModule == 'HammingEqOdd':
            self.fairnessModule = HammingEqualizedOdds.HammingEqualizedOdds(self.args)
        else:
            raise Exception('No associated fairness module found.')

