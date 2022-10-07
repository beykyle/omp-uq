import sys
import os
from pathlib import Path

from cgmf_uq import process_ensemble

# where are we putting the results
res_dir = Path("/home/beykyle/turbo/omp_uq/KDUQ/post_proc")
hist_dir = Path("/home/beykyle/turbo/omp_uq/KDUQ/histories_192E3")

for i in range(11,416):
    process_ensemble(hist_dir, res_dir, str(i))
