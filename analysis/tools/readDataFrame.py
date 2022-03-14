import pandas as pd
from numpy.random import multivariate_normal

parameters = pd.io.json.read_json("parametersDataFrame.json", orient='split')
covariance = parameters.cov()

# modify to your desired number of samples
nSamples = 100
samples = multivariate_normal(parameters.mean(), parameters.cov(), size=nSamples)

