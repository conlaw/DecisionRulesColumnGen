from scipy.stats import bernoulli
import numpy as np
from RuleSampler import RuleSampler

class TopXRuleSampler(RuleSampler):
    
    def __init__(self, args = {}):
        self.numRulesToReturn = args['numRulesToReturn'] if 'numRulesToReturn' in args else 100
        print("hey your config said to return this many rules: ", self.numRulesToReturn)
    
    def getRules(self, rules, reduced_costs, col_samples):
        sorted_rc = np.argsort(reduced_costs)
        
        final_rules = []
        rules = np.array(rules)
        for rule in sorted_rc[:self.numRulesToReturn]:
            fin_rule = np.zeros(len(col_samples))
            fin_rule[col_samples] = rules[rule]
            final_rules.append(fin_rule)

        return final_rules, np.array(reduced_costs)[sorted_rc[:self.numRulesToReturn]]