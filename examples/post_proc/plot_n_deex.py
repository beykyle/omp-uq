import sys
from pathlib import Path
from CGMFtk import histories as fh
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

file = Path(sys.argv[1])

assert(file.is_file())

print("Reading from {}".format(file))
h = fh.Histories(file, ang_mom_printed=True)

df = h.getParticleEventDataFrame()

g = sns.jointplot(
    data=df,
    x="nEcm", y="cmNdJ", hue="neutron_order",
    kind="kde",
)

plt.xlabel(r"$\Delta E$ [MeV]")
plt.xlabel(r"$\Delta J$ [MeV]")
plt.show(fig)
