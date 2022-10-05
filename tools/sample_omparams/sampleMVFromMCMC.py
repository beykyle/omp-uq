import pandas as pd
from numpy.random import multivariate_normal

def samplesFromNormal(mcmc_sample_file, nSamples):
    parameters = pd.io.json.read_json(mcmc_sample_file, orient='split')
    covariance = parameters.cov()
    samples = multivariate_normal(parameters.mean(), parameters.cov(), size=nSamples)
    return samples

