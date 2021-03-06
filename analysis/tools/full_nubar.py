import sys
import numpy as np
from pathlib import Path

sys.path.append("/home/beykyle/omp-uq/analysis/tools")

from run_ensemble import calculate_ensemble_nubar

results_dir = Path(sys.argv[1])
nubars = []

nsamples = 415


for i in range(0,nsamples+1):
    nubars.append( calculate_ensemble_nubar(str(i),results_dir) )

np.save(str(results_dir / "nubars"), np.array(nubars))
