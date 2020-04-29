import numpy as np
import pandas as pd
from binerizer import *
from DNFRuleModel import DNFRuleModel
from MasterModel import MasterModel
from RuleGenerator import RuleGenerator
from Classifier import Classifier
from sklearn.model_selection import train_test_split
import random
import time

def run_test(args, verbose = False):
    #Do Sanity Testing on inputs
    print(args)
    if 'hyper_paramaters' not in args:
        print('No hyper parameters input. Running model with defaults!')
    else:
        for hp in args['hyper_paramaters']:
            print('%d parameters provided for hyper-parameter %s'%(len(args['hyper_paramaters'][hp]),hp))
    
    if 'data' not in args:
        print('No data file path provided. Exiting.')
        return
    
    fixedModelParams = args['fixed_model_params'] if 'fixed_model_params' in args else {}
    price_limit = args['price_limit'] if 'price_limit' in args else 999999999
    train_limit = args['train_limit'] if 'train_limit' in args else 999999999
    num_splits = args['num_splits'] if 'num_splits' in args else 2
    num_splits_hp = args['num_splits_hp'] if 'num_splits_hp' in args else 2
    test_name = args['test_name'] if 'test_name' in args else args['data']
    
    #Read and prepare data
    print('Reading in data')
    
    data = pd.read_csv(args['data'])
    data = binerizeData(data, verbose = verbose)
    data = data.sample(frac=1).reset_index(drop=True)

    data_X = data.to_numpy()[:,0:(data.shape[1]-1)]
    data_Y = data.to_numpy()[:,data.shape[1]-1]

    #Prepare results dict
    results = {}
    
    accuracies = []
    mip = []
    mip_final = []
    ip = []
    complexities = []
    times = []
    overall_num_iter = []
    overall_num_cols_gen = []
    total_rules = []
    
    
    #Prepare data indices
    break_points = np.floor(np.arange(0,1+1/num_splits,1/num_splits)*data_X.shape[0]).astype(np.int)

    for i in range(num_splits):
        print('Running split %d'%i)
        #Get data for this fold
        X_train = np.delete(data_X, np.arange(break_points[i], break_points[i+1]), axis=0)
        Y_train = np.delete(data_Y, np.arange(break_points[i], break_points[i+1]))

        X_test = data_X[np.arange(break_points[i], break_points[i+1]),:]
        Y_test = data_Y[np.arange(break_points[i], break_points[i+1])]
        
        saved_rules = None
        
        break_points_hp = np.floor(np.arange(0,1+1/num_splits_hp,1/num_splits_hp)*X_train.shape[0]).astype(np.int)
        hp_results = {}

        for hp in args['hyper_paramaters']:
            for hp_val in args['hyper_paramaters'][hp]:
                print('Testing %s value %d'%(hp, hp_val))
                
                fixedModelParams[hp] = hp_val
                res = {}
                res['hp_val'] = hp_val
                
                hp_accuracies = []    
                hp_complexities = []
                hp_times = []
                hp_num_iter = []
                hp_num_cols_gen = []
                hp_mip = []
                hp_mip_final = []
                hp_ip = []

                hp_rules = None
                
                for j in range(num_splits_hp):
                    print('Split %d'%j)
                    X_hp_train = np.delete(X_train, np.arange(break_points_hp[j], break_points_hp[j+1]), axis = 0)
                    Y_hp_train = np.delete(Y_train, np.arange(break_points_hp[j], break_points_hp[j+1]))

                    X_hp_test = X_train[np.arange(break_points_hp[j], break_points_hp[j+1]),:]
                    Y_hp_test = Y_train[np.arange(break_points_hp[j], break_points_hp[j+1])]
                    
                    
                    classif = Classifier(X_hp_train, Y_hp_train, fixedModelParams, ruleGenerator = 'Generic')
                    
                    start_time = time.perf_counter() 
                    classif.fit(initial_rules = hp_rules,
                                verbose = False, 
                                timeLimit = train_limit, 
                                timeLimitPricing = price_limit)
                    time_to_exec = time.perf_counter() - start_time

                    #Get results of test
                    hp_times.append(time_to_exec)
                    
                    acc = sum(classif.predict(X_hp_test) == Y_hp_test)/len(Y_hp_test)
                    hp_accuracies.append(acc)
                    
                    complexity = np.sum(classif.fitRuleSet)+ len(classif.fitRuleSet)
                    hp_complexities.append(complexity)
                    
                    num_iter = classif.numIter
                    hp_num_iter.append(num_iter)
                    
                    num_cols_gen = len(classif.ruleMod.rules)
                    hp_num_cols_gen.append(num_cols_gen)
                    
                    hp_mip.append(classif.mip_results)
                    hp_mip_final.append(classif.final_mip)
                    hp_ip.append(classif.final_ip)

                    print('Results: (Acc) %.3f (Time) %.0f (Complex) %.0f (Cols Gen) %.0f  (Num Iter) %.0f'%(acc, time_to_exec, complexity, num_cols_gen, num_iter))

                    #Save any rules generated to use in final column gen prediction process
                    rules = classif.ruleMod.rules
                    if saved_rules is None:
                        saved_rules = rules
                    else:
                        saved_rules = np.unique(np.concatenate([saved_rules,rules]), axis = 0)
                        
                    if hp_rules is None:
                        hp_rules = rules
                    else:
                        hp_rules = np.unique(np.concatenate([hp_rules,rules]), axis = 0)
                
                res['mean_accuracy'] = np.mean(hp_accuracies)
                res['sd_accuracy'] = np.std(hp_accuracies)
                res['accuracies'] = hp_accuracies
                
                res['mean_complexity'] = np.mean(hp_complexities)
                res['sd_complexity'] = np.std(hp_complexities)
                res['complexities'] = hp_complexities
                
                res['mean_time'] = np.mean(hp_times)
                res['sd_time'] = np.std(hp_times)
                res['time'] = hp_times

                res['mean_cols'] = np.mean(hp_num_cols_gen)
                res['sd_cols'] = np.std(hp_num_cols_gen)
                res['cols'] = hp_num_cols_gen
                
                res['mean_iter'] = np.mean(hp_num_iter)
                res['sd_iter'] = np.std(hp_num_iter)
                res['iter'] = hp_num_iter
                
                res['mean_final_mip'] = np.mean(hp_mip_final)
                res['sd_final_mip'] = np.std(hp_mip_final)
                res['final_mip'] = hp_mip_final
                
                res['mean_ip'] = np.mean(hp_ip)
                res['sd_ip'] = np.std(hp_ip)
                res['ip'] = hp_ip
                
                res['mip'] = hp_mip

                print('HP Results: (Acc) %.3f (Time) %.0f (Complex) %.0f (Cols Gen) %.0f  (Num Iter) %.0f'%(np.mean(hp_accuracies), 
                                                                                                            np.mean(hp_times), 
                                                                                                            np.mean(hp_complexities), 
                                                                                                            np.mean(hp_num_cols_gen), 
                                                                                                            np.mean(hp_num_iter)))

                hp_results[str(hp)+'-'+str(hp_val)] = res

            optimal_param_value = [hp_results[x]['hp_val'] for x in hp_results][np.argmax([hp_results[x]['mean_accuracy'] for x in hp_results])]
            fixedModelParams[hp] = optimal_param_value
            
        print('Running final model with parameters: '+ str(fixedModelParams))
            
        classif = Classifier(X_train, Y_train, fixedModelParams, ruleGenerator = 'Generic')
                    
        start_time = time.perf_counter() 
        classif.fit(initial_rules = saved_rules, 
                    verbose = False, 
                    timeLimit = train_limit, 
                    timeLimitPricing = price_limit)
        time_to_exec = time.perf_counter() - start_time

        #Get results of test
        times.append(time_to_exec)
                    
        acc = sum(classif.predict(X_test) == Y_test)/len(Y_test)
        accuracies.append(acc)
                    
        complexity = np.sum(classif.fitRuleSet)+ len(classif.fitRuleSet)
        complexities.append(complexity)
                    
        num_iter_fold = classif.numIter
        overall_num_iter.append(num_iter_fold)
                    
        num_cols_gen = len(classif.ruleMod.rules) - len(saved_rules)
        overall_num_cols_gen.append(num_cols_gen)
            
        total_rule = len(classif.ruleMod.rules)
        total_rules.append(total_rule)
        
        mip.append(classif.mip_results)
        mip_final.append(classif.final_mip)
        ip.append(classif.final_ip)

        print('Overall Fold Results: (Acc) %.3f (Time) %.0f (Complex) %.0f (Cols Gen) %.0f  (Num Iter) %.0f (Total rules) %.0f'%(acc, 
                                                                                                            time_to_exec, 
                                                                                                            complexity, 
                                                                                                            num_cols_gen, 
                                                                                                            num_iter_fold,
                                                                                                            total_rule))
    
        results['fold-%d-hp'%i] = hp_results

    results['mean_accuracy'] = np.mean(accuracies)
    results['sd_accuracy'] = np.std(accuracies)
    results['accuracies'] = accuracies
                
    results['mean_complexity'] = np.mean(complexities)
    results['sd_complexity'] = np.std(complexities)
    results['complexities'] = complexities
                
    results['mean_time'] = np.mean(times)
    results['sd_time'] = np.std(times)
    results['time'] = times

    results['mean_cols'] = np.mean(overall_num_cols_gen)
    results['sd_cols'] = np.std(overall_num_cols_gen)
    results['cols'] = overall_num_cols_gen
                
    results['mean_iter'] = np.mean(overall_num_iter)
    results['sd_iter'] = np.std(overall_num_iter)
    results['iter'] = overall_num_iter
    
    results['mean_total_rules'] = np.mean(total_rules)
    results['sd_total_rules'] = np.std(total_rules)
    results['total_rules'] = total_rules
    
    results['mean_final_mip'] = np.mean(mip_final)
    results['sd_final_mip'] = np.std(mip_final)
    results['final_mip'] = mip_final
                
    results['mean_ip'] = np.mean(ip)
    results['sd_ip'] = np.std(ip)
    results['ip'] = ip
                
    results['mip'] = mip
    
    print('Final Results: (Acc) %.3f (Time) %.0f (Complex) %.0f (Cols Gen) %.0f  (Num Iter) %.0f (Total rules) %.0f'%(np.mean(accuracies), 
                                                                                                            np.mean(times), 
                                                                                                            np.mean(complexities), 
                                                                                                            np.mean(overall_num_cols_gen), 
                                                                                                            np.mean(overall_num_iter),
                                                                                                            np.mean(total_rules)))


    return results     