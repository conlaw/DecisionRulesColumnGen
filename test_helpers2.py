import numpy as np
import pandas as pd
from binerizer import *
from DNFRuleModel import DNFRuleModel
from Classifier import Classifier
from sklearn.model_selection import train_test_split
import random
import time
import json
import copy
class TestResults(object):
    '''
    Object to contain and run the restricted model
    '''
    def __init__(self, testName, results_path = './results/'):
        self.name = testName
        self.resultsPath = results_path
        self.res = {}
        self.metrics = ['accuracy',
                       'mip',
                       'mip_final',
                       'ip',
                       'complexity',
                       'fairness',
                       'times',
                       'num_iter',
                       'num_cols_gen']
        
        for metric in self.metrics:
            self.res[metric] = []
        
    def update(self, classifier, time, test_data):
        self.res['times'].append(time)
        self.res['accuracy'].append(sum(classifier.predict(test_data[0]) == test_data[1])/len(test_data[1]))
        self.res['complexity'].append(np.sum(classifier.fitRuleSet)+ len(classifier.fitRuleSet))
        self.res['num_iter'].append(classifier.numIter)
        self.res['num_cols_gen'].append(len(classifier.ruleMod.rules))
        self.res['mip'].append(classifier.mip_results)
        self.res['mip_final'].append(classifier.final_mip)
        self.res['ip'].append(classifier.final_ip)
        
        for metric in self.metrics:
            if metric == 'mip':
                continue
            self.res['mean_'+metric] = np.mean(self.res[metric])
            self.res['std_'+metric] = np.std(self.res[metric])
        
        self.printResult()
        self.write()

    def printResult(self):
        print('Results: (Acc) %.3f (Time) %.0f (Complex) %.0f (Cols Gen) %.0f  (Num Iter) %.0f'%(self.res['accuracy'][-1], 
                                                                                                 self.res['times'][-1], 
                                                                                                 self.res['complexity'][-1], 
                                                                                                 self.res['num_cols_gen'][-1], 
                                                                                                 self.res['num_iter'][-1]))
    def write(self):
        print(self.res)
        r = json.dumps(self.res)
        
        with open(self.resultsPath+self.name+".txt", "w") as text_file:
            text_file.write(r)


def checkArgs(args):
    #Do Sanity Testing on inputs
    print(args)
    if 'hyper_paramaters' not in args:
        print('No hyper parameters input. Running model with defaults!')
    else:
        for hp in args['hyper_paramaters']:
            print('%d parameters provided for hyper-parameter %s'%(len(args['hyper_paramaters'][hp]),hp))
    

def extractArgs(args):
    new_args = {}
    new_args['fixed_model_params'] = args['fixed_model_params'] if 'fixed_model_params' in args else {}
    new_args['price_limit'] = args['price_limit'] if 'price_limit' in args else 999999999
    new_args['train_limit'] = args['train_limit'] if 'train_limit' in args else 999999999
    new_args['num_hp_splits'] = args['num_hp_splits'] if 'num_hp_splits' in args else 3
    new_args['name'] = args['name'] 
    new_args['hyper_paramaters'] = args['hyper_paramaters']
    
    return new_args

def extractGlobalArgs(args):
    new_args = {}
    new_args['num_splits'] = args['num_splits'] if 'num_splits' in args else 10
    return new_args

def readData(dataInfo):
    data = pd.read_csv(dataInfo['file'])
    data = binerizeData(data)
    data = data.sample(frac=1).reset_index(drop=True)

    cols = data.columns
    group_index = [i for i in range(len(cols)) if cols[i] == dataInfo['groupCol']][0]
    data_X = data.to_numpy()[:,0:(data.shape[1]-1)]
    data_Y = data.to_numpy()[:,data.shape[1]-1]
    group = data.to_numpy()[:,group_index]
    
    return data_X, data_Y, group

def getFold(X,Y,group, indices):
    X_train = np.delete(X, indices, axis=0)
    Y_train = np.delete(Y, indices)
    g_train = np.delete(group, indices)

    X_test = X[indices,:]
    Y_test = Y[indices]
    
    return X_train, Y_train, g_train, X_test, Y_test

def updateRuleSet(rule_set, rules):
    if rule_set is None:
        rule_set = rules
    else:
        rule_set = np.unique(np.concatenate([rule_set,rules]), axis = 0)
    
    return rule_set

def runSingleTest(X_tr, Y_tr, g_tr, X_t, Y_t, test_params, rules, res):
    test_params['fixed_model_params']['group'] = g_tr

    classif = Classifier(X_tr, Y_tr, test_params['fixed_model_params'], ruleGenerator = 'Generic')

    start_time = time.perf_counter() 
    classif.fit(initial_rules = rules,
                verbose = False, 
                timeLimit = test_params['train_limit'], 
                timeLimitPricing = test_params['price_limit'])
    time_to_exec = time.perf_counter() - start_time
                        
    res.update(classif, time_to_exec, (X_t, Y_t))
    
    return res, classif

def runNestedCV(X, Y, group, test_params, foldId = -1):
    saved_rules = None
    hp_results = {}#init results object
    break_points_hp = np.floor(np.arange(0,1+1/test_params['num_hp_splits'],
                                         1/test_params['num_hp_splits'])*X.shape[0]).astype(np.int)

    for hp in test_params['hyper_paramaters']:
        for hp_val in test_params['hyper_paramaters'][hp]:

            res = TestResults(test_params['name']+' '+'(%s,%d)'%(hp, hp_val)+'-'+str(foldId))
            test_params['fixed_model_params'][hp] = hp_val
            res.res['hp'] = hp
            res.res['hp_val'] = hp_val

            hp_rules = None

            for j in range(test_params['num_hp_splits']):
                print('Split %d'%j)
                #Get fold
                X_tr, Y_tr, g_tr, X_t, Y_t = getFold(X,Y, group, np.arange(break_points_hp[j], break_points_hp[j+1]))
                res, classif = runSingleTest(X_tr, Y_tr, g_tr, X_t, Y_t, test_params, hp_rules, res)
                        
                #Save any rules generated to use in final column gen prediction process
                rules = classif.ruleMod.rules
                saved_rules = updateRuleSet(saved_rules,rules)
                hp_rules = updateRuleSet(hp_rules,rules)
                    
            #Store quick access for accuracy mean result
            hp_results[str(hp_val)] = res.res['mean_accuracy']
                
        #Get optimal HP val
        optimal_param_value = [float(x) for x in hp_results][np.argmax([hp_results[x] for x in hp_results])]
        test_params['fixed_model_params'][hp] = optimal_param_value
    
    return test_params, saved_rules

def run_test(tests_raw, globalArgs, dataInfo, save_rule_set = False, results_path = './results/', verbose = False):
    globalRules = None
    globArgs = extractGlobalArgs(globalArgs)
    
    #Set-up Data
    X,Y,group = readData(dataInfo)
                
    #Set-up tests
    tests = []
    for test in tests_raw:
        checkArgs(test)
        tests.append(extractArgs(test))
    
    #Create results
    res = []
    for i in range(len(tests)):
        res.append(TestResults(tests[i]['name']))
        
    #Prepare data indices
    break_points = np.floor(np.arange(0,1+1/globArgs['num_splits'],1/globArgs['num_splits'])*X.shape[0]).astype(np.int)

    for i in range(globArgs['num_splits']):
        print('****** Running split %d ******'%i)
        
        #Get data for this fold
        X_train, Y_train, g_train, X_test, Y_test = getFold(X,Y, group, np.arange(break_points[i], break_points[i+1]))
        
        #Run every test for this fold
        for test in range(len(tests)):
            print('**** Running Test %s ****'%tests[test]['name'])
            final_params, saved_rules = runNestedCV(X_train, Y_train, g_train, tests[test], foldId = i)
            
            globalRules = updateRuleSet(globalRules, saved_rules)
            
            if save_rule_set:
                r = json.dumps(globalRules.tolist())
                with open(results_path+tests[test]['name']+'_rules.txt', "w") as text_file:
                    text_file.write(r)

            print('Running final model with parameters: '+ str(final_params['fixed_model_params']))
            res[test], _ = runSingleTest(X_train, Y_train, g_train, X_test, Y_test, final_params, saved_rules, res[test])

    return res, globalRules