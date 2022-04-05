import numpy as np
import os
import re
from pathlib import Path
from CGMFtk import histories as fh
import sys

sys.path.append("/home/beykyle/omp-uq/analysis/tools")

import run_ensemble

nevents = int(2E3)

# only use default params
param_fname = Path("/home/beykyle/OM/KDOMPuq/KDGlobal.json")
results_dir = Path("/home/beykyle/scratch/omp_uq/cf_252_sf/default/out_192kh_415s")

zaid = 98252
energy_MeV = 0.0

for i in range(287,416):
    sample_name = str(i)
    run_ensemble.run_ensemble(param_fname, results_dir, nevents, zaid, energy_MeV, sample_name)
    run_ensemble.process_ensemble(results_dir, sample_name)
    run_ensemble.process_ensemble_corr(results_dir, sample_name)
    np.save(results_dir / "nubars", np.array([run_ensemble.calculate_ensemble_nubar(sample_name, results_dir)]))
