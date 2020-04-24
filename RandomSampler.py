from scipy.stats import bernoulli
import numpy as np
from SubSampler import SubSampler

class RandomSampler(SubSampler):
    
    def __init__(self, args = {}):
        self.rowSample = args['rowSample'] if 'rowSample' in args else True
        self.columnSample = args['colSample'] if 'colSample' in args else True
        self.samplePercRow = args['samplePercRow'] if 'samplePercRow' in args else 2000
        self.samplePercCol = args['samplePercCol'] if 'samplePercCol' in args else 100000

    
    def getSample(self, X, Y, mu, args = {}):
       
        X_sample = X
        Y_sample = Y
        mu_sample = np.array(mu)
        col_samples = np.ones(X.shape[1]).astype(np.bool)
        
        if self.rowSample:
            perc = min(self.samplePercRow/X.shape[0],1)
            row_samples = bernoulli.rvs(perc, size=X.shape[0]).astype(np.bool)
            X_sample = X_sample[row_samples, :]
            Y_sample = Y_sample[row_samples]
            mu_sample = mu_sample[row_samples[Y]]
        if self.columnSample:
            col_perc = min(self.samplePercCol/np.mean(np.sum(X > 0, axis=0))/X.shape[1],1)
            col_samples = bernoulli.rvs(col_perc, size=X.shape[1]).astype(np.bool)
            X_sample = X_sample[:, col_samples]
        
        return X_sample, Y_sample, mu_sample, col_samples      
