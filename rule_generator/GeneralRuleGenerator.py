import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .data_sampler import *
from .generator import *
from .rule_sampler import *
import time

class GeneralRuleGenerator(object):
    '''
    Object to create a binary classifier using column generation framework
    '''
    
    def __init__(self, ruleMod, fairnessModule,
                 args = {},
                 sampler = 'random',
                 ruleGenerator = 'Greedy',
                 ruleSelect = 'topX'):
        
        self.args = args
        self.ruleMod = ruleMod
        self.fairnessModule = fairnessModule
        
        ruleGenerator = args['ruleGenerator'] if 'ruleGenerator' in args else ruleGenerator
        print('Using %s'%ruleGenerator)
        
        #Define class variables
        self.initSampler(sampler)
        self.initRuleGenerator(ruleGenerator)
        self.initRuleSelect(ruleSelect)
   
   
    def generateRule(self, args = {}):
        
        if 'lam' not in args or 'mu' not in args or 'alpha' not in args:
            raise Exception('Required arguments not supplied for DNF IP Rule Generator.')
        
        #Sample Datasets
        X, Y, args['mu'], args['alpha'], args['row_samples'], col_samples = self.sampler.getSample(self.ruleMod.X, 
                                                                                                   self.ruleMod.Y, 
                                                                                                   args['mu'], args['alpha'])
        #Generate Rules
        rules, rcs = self.ruleGen.generateRule(X, Y, args)
        
        #Subsample rules to return
        final_rules, final_rcs = self.ruleSelect.getRules(rules, rcs, col_samples)

        return len(final_rules) > 0 , final_rules
        
        
                
    def initSampler(self, sampler):
        '''
        Function that maps string rule models to objects
           - To add a new rule model simply add the object to the if control flow
        '''
        
        if sampler == 'full':
            self.sampler = notsosubSampler.NotSoSubSampler()
        elif sampler == 'random':
            self.sampler = RandomSampler.RandomSampler(self.args)
        else:
            raise Exception('No associated rule model found.')
    
    def initRuleGenerator(self, ruleGenerator):
        '''
        Function that maps string rule generators to objects
           - To add a new rule generator simply add the object to the if control flow
        '''

        if ruleGenerator == 'DNF_IP':
            self.ruleGen = DNF_IP_RuleGenerator.DNF_IP_RuleGenerator(self.fairnessModule, self.args)
        elif ruleGenerator == 'DNF_IP_OPT':
            self.ruleGen = DNF_IP_RuleGeneratorOpt.DNF_IP_RuleGeneratorOpt(self.fairnessModule, self.args)
        elif ruleGenerator == 'Greedy':
            self.ruleGen = GreedyRuleGenerator.GreedyRuleGenerator(self.fairnessModule, self.args)
        elif ruleGenerator == 'Hybrid':
            self.ruleGen = HybridGenerator.HybridGenerator(self.fairnessModule, self.args)
        else:
            raise Exception('No associated rule generator found.')
            
    def initRuleSelect(self, ruleSelect):
        '''
        Function that maps string rule selection rules to objects
           - To add a new rule selection rule simply add the object to the if control flow
        '''
        if ruleSelect == 'full':
            self.ruleSelect = FullRuleSampler.FullRuleSampler(self.args)
        elif ruleSelect == 'topX':
            self.ruleSelect = TopXRuleSampler.TopXRuleSampler(self.args)
        elif ruleSelect == 'random':
            self.ruleSelect = NaifRandomRuleSampler.NaifRandomRuleSampler(self.args)
        elif ruleSelect == 'softmax':
            self.ruleSelect = SoftmaxRandomRuleSampler.SoftmaxRandomRuleSampler(self.args)
        else:
            raise Exception('No associated rule selector found.')

            
        
