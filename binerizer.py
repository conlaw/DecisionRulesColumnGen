import numpy as np
import pandas as pd

'''
Helper functions to binerize data columns
'''

def binNumeric(data, column, quantiles = 10):
    '''
    Bins numeric columns by quantile (optional parameter)
    Returns: Pandas dataframe with one column per bin
    '''
    return pd.get_dummies(pd.qcut(data[column].values, quantiles), prefix = column)
    
def binCategorical(data, column):
    '''
    Creates one hot encoding for categorical variables
    Returns: Pandas dataframe with one column per bin
    '''
    return pd.get_dummies(data[column], prefix = column)
    
def threshNumeric(data, column, quant = 10):
    '''
    Creates binarized numeric features using sequence of thresholds specified by the sample quantiles
    Returns: Pandas dataframe with one column representing > quant and one <= quant for each sample quantile
    '''
    quantiles = np.quantile(data[column], np.arange(1/quant,1,1/quant))
    lowThresh = np.transpose([np.where(data[column] <= x, 1, 0) for x in quantiles])
    highThresh = 1-lowThresh
    
    binarized = pd.DataFrame(np.concatenate((lowThresh, highThresh), axis = 1))
    binarized.columns = [column+'_'+str(round(x,2))+'*' for x in quantiles] + [column+'_'+str(round(x,2))+'_' for x in quantiles]

    return binarized



                            