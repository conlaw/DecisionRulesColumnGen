import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from notsosubSampler import NotSoSubSampler
from RandomSampler import RandomSampler
from DNF_IP_RuleGenerator2 import DNF_IP_RuleGenerator2
from GreedyRuleGenerator import GreedyRuleGenerator
from FullRuleSampler import FullRuleSampler
from TopXRuleSampler import TopXRuleSampler
from NaifRandomRuleSampler import NaifRandomRuleSampler
from SoftmaxRandomRuleSampler import SoftmaxRandomRuleSampler
from RuleGenerator import RuleGenerator

class GeneralRuleGenerator(object):
    '''
    Object to create a binary classifier using column generation framework
    '''
    
    def __init__(self, ruleMod,
                 args = {},
                 sampler = 'random',
                 ruleGenerator = 'Greedy',
                 ruleSelect = 'topX'):
        
        self.args = args
        self.ruleMod = ruleMod
        
        ruleGenerator = args['ruleGenerator'] if 'ruleGenerator' in args else ruleGenerator
        print('Using %s'%ruleGenerator)
        
        #Define class variables
        self.initSampler(sampler)
        self.initRuleGenerator(ruleGenerator)
        self.initRuleSelect(ruleSelect)
   
   
    def generateRule(self, args = {}):
        
        if 'lam' not in args or 'mu' not in args:
            raise Exception('Required arguments not supplied for DNF IP Rule Generator.')

        lam = args['lam']
        mu = args['mu']
        
        X, Y, mu, col_samples = self.sampler.getSample(self.ruleMod.X, self.ruleMod.Y, mu)
        rules, rcs = self.ruleGen.generateRule(X, Y, lam, mu, args)
        final_rules, final_rcs = self.ruleSelect.getRules(rules, rcs, col_samples)
        
        return len(final_rules) > 0 , final_rules
        
        
                
    def initSampler(self, sampler):
        '''
        Function that maps string rule models to objects
           - To add a new rule model simply add the object to the if control flow
        '''
        
        if sampler == 'full':
            self.sampler = NotSoSubSampler()
        elif sampler == 'random':
            self.sampler = RandomSampler(self.args)
        else:
            raise Exception('No associated rule model found.')
    
    def initRuleGenerator(self, ruleGenerator):
        '''
        Function that maps string rule generators to objects
           - To add a new rule generator simply add the object to the if control flow
        '''

        if ruleGenerator == 'DNF_IP':
            self.ruleGen = DNF_IP_RuleGenerator2(self.args)
        elif ruleGenerator == 'Greedy':
            self.ruleGen = GreedyRuleGenerator(self.args)
        else:
            raise Exception('No associated rule generator found.')
            
    def initRuleSelect(self, ruleSelect):
        
        if ruleSelect == 'full':
            self.ruleSelect = FullRuleSampler(self.args)
        elif ruleSelect == 'topX':
            self.ruleSelect = TopXRuleSampler(self.args)
        elif ruleSelect == 'random':
            self.ruleSelect = NaifRandomRuleSampler(self.args)
        elif ruleSelect == 'softmax':
            self.ruleSelect = SoftmaxRandomRuleSampler(self.args)
        else:
            raise Exception('No associated rule selector found.')

            
        
