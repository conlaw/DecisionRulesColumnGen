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
        
        if 'lam' not in args or 'mu' not in args:
            raise Exception('Required arguments not supplied for DNF IP Rule Generator.')
        print('Starting row sampling')
        start = time.perf_counter()
        X, Y, args['mu'], args['row_samples'], col_samples = self.sampler.getSample(self.ruleMod.X, self.ruleMod.Y, args['mu'])
        print('Row sampling took %.2f seconds'%(time.perf_counter() - start))
        
        print('Starting Rule generation')
        start = time.perf_counter()
        rules, rcs = self.ruleGen.generateRule(X, Y, args)
        print('Rule generation took %.2f seconds'%(time.perf_counter() - start))

        print('Starting Rule selection')
        start = time.perf_counter()
        final_rules, final_rcs = self.ruleSelect.getRules(rules, rcs, col_samples)
        print('Rule selection took %.2f seconds'%(time.perf_counter() - start))

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
        elif ruleGenerator == 'Greedy':
            self.ruleGen = GreedyRuleGenerator.GreedyRuleGenerator(self.fairnessModule, self.args)
        else:
            raise Exception('No associated rule generator found.')
            
    def initRuleSelect(self, ruleSelect):
        
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

            
        
