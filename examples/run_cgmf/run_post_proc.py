import sys
import os
from pathlib import Path

from cgmf_uq import process_ensemble, calculate_ensemble_nubar

# where are we putting the results
res_dir = Path("/home/beykyle/turbo/omp_uq/KDUQ/u235/post_proc")
hist_dir = Path("/home/beykyle/turbo/omp_uq/KDUQ/u235/histories_192E3")

# default
nubars = []
default_dir = res_dir / "default"
np.save(str(default_dir / "nubars"), np.array(nubars))
nubars.append( calculate_ensemble_nubar("default", default_dir) )
np.save(str(default_dir / "nubars_default"), np.array(nubars))

# uq
nubars = []
for i in range(0,416):
    process_ensemble(hist_dir, res_dir, str(i))
    nubars.append( calculate_ensemble_nubar(str(i),results_dir) )

np.save(str(results_dir / "nubars"), np.array(nubars))

