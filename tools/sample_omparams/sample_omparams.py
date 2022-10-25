import pandas as pd
from numpy.random import multivariate_normal
import sys
import json
from pathlib import Path

def generate_samples(df, nSamples):
    covariance = df.cov()
    samples = multivariate_normal(df.mean(), covariance, size=nSamples)
    return samples

def print_to_individual_files(df, samples, outpath, name):
    c = list(df.columns)
    for i, s in enumerate(samples):
        out = pd.DataFrame([s], columns=c)
        out_path = outpath / Path(str(name) + "_{}.json".format(i))
        parsed = json.loads(out.to_json(orient="records"))
        with open(out_path, "w") as o:
            json.dump(parsed[0], o, indent=2)

    # handle default
    out = pd.DataFrame([df.mean()], columns=c)
    out_path = outpath / Path(str(name) + "_default.json")
    parsed = json.loads(out.to_json(orient="records"))
    with open(out_path, "w") as o:
        json.dump(parsed[0], o, indent=2)


if __name__ == "__main__":
    df = pd.io.json.read_json(sys.argv[1], orient='split')
    s = generate_samples(df, int(sys.argv[3]))
    print_to_individual_files(df, s, Path("./"), Path(sys.argv[2]))
