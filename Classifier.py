import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from DNFRuleModel import DNFRuleModel
from MasterModel import MasterModel
from DNF_IP_RuleGenerator import DNF_IP_RuleGenerator

class Classifier(object):
    '''
    Object to create a binary classifier using column generation framework
    '''
    
    def __init__(self, X, Y,
                 ruleModel = 'DNF',
                 ruleGenerator = 'DNF_IP'):
        
        #Define class variables
        self.ruleMod = None
        self.ruleGen = None
        self.master = None
        self.fitRuleSet = None
        
        # Map parameters to instantiated objects
        self.initRuleModel(X, Y, ruleModel)
        self.initRuleGenerator(ruleGenerator)
        self.master = MasterModel(self.ruleMod)
        
    def fit(self, initial_rules = None, verbose = False):
        '''
        Function to generate a rule set
            - Can take initial set of rules
            - Verbose parameter controls how much output is displayed during intermediary steps
        '''
        
        # Add initial rules to master model
        if initial_rules is not None:
            self.master.addRules(initial_rules)
        
        while True:
            
            # Solve relaxed version of restricted problem
            if verbose:
                print('Solving Restricted LP')
            results = self.master.solve(verbose = verbose, relax = True)
            results['verbose'] = verbose
            
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
                break
        
        # Solve master problem to optimality
        if verbose:
            print('Solving final master problem to integer optimality')
        results = self.master.solve(verbose = verbose, relax = False)
        
        self.fitRuleSet = results['ruleSet']
        
        #Return final rules
        return self
        
    def predict(self, X):
        '''
        Function to predict class labels using the fitted rule set
        '''
        if self.fitRuleSet is None:
            raise Exception("Model not fit. Can't make inference!")
        
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

        if ruleGenerator == 'DNF_IP':
            self.ruleGen = DNF_IP_RuleGenerator(self.ruleMod)
        else:
            raise Exception('No associated rule generator found.')
        
