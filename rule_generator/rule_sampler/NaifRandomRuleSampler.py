from scipy.stats import bernoulli
import numpy as np
from .RuleSampler import RuleSampler

class NaifRandomRuleSampler(RuleSampler):
    
    def __init__(self, args = {}):
        self.numRulesToReturn = args['numRulesToReturn'] if 'numRulesToReturn' in args else 20
    
    def getRules(self, rules, reduced_costs, col_samples):
        returnNum = min(self.numRulesToReturn, len(rules))
            
        rules_to_return = np.random.choice(range(len(rules)), returnNum, replace = False)
        
        final_rules = []
        rules = np.array(rules)
        for rule in rules_to_return:
            fin_rule = np.zeros(len(col_samples))
            fin_rule[col_samples] = rules[rule]
            final_rules.append(fin_rule)
            

        return final_rules, np.array(reduced_costs)[rules_to_return]