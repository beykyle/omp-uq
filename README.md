# omp-uq

Uncertainty quantification of fission observables calculated with a fission event generator with Monte Carlo Hauser-Fresbach fragment de-excitation, due to phenomenological and theoretical uncertainties in various optical models used for neutron emission from fragments. One example is the [Koning-Delaroche global optical model](https://www.sciencedirect.com/science/article/pii/S0375947402013210?casa_token=ADeCX1nO83AAAAAA:Xwa6DlMKYvVU0ygGxoD0C6bfFlG0UB9hrOHojDbv2dQ7zsZvd7hhlZzvDo1b1sVxOYzL90kj), which contains model parameter uncertainty from fitting to scattering observables. Another example is the [WLH model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.182502), which contains various sources of theoretical uncertainty, from truncating the expansion of nucleon-nucleon forces in chiral effective field theory, from reducing to an effective 1-body potential with many-body perturbation theory, and from the transition from nuclear matter to finite nuclei with the improved local density approximation. By comparing various phenomenological and theoretical models with UQ, one can test how well these models are at describing the single particle properties of neutron rich, highly excited fission fragments. In particular, the extrapolation of phenomenological models to the neutron rich region, and the predictive power of microscopic models in this region, is interesting. 

This code takes in a covariance matrix or distribution of global optical model parameters, and performs UQ on a variety of fission observables; for model to experiment comparisons, and non-observables; for model-model comparison.  This is done by propagating parameter uncertanties through a modified version of LANL's [CGMF](https://github.com/lanl/cgmf) fission event generator. The modified version used in this code lives [here](https://github.com/beykyle/cgmf), and is included as a git submodule.

The meat of this code is the python modules in [tools/](https://github.com/beykyle/omp-uq/tree/main/tools/) that run CGMF and analyze histories.

# download 

```
git clone --recurse-submodules git@github.com:beykyle/omp-uq.git
```

# install

First, [build and install](https://github.com/beykyle/CGMF/#quickstart) the linked submodule verison of CMGF, for example:

```
cd CGMF
py setup.py build -j{nproc}
cd ..
pip install -e CGMF --user
```

Then install the packages in `./tools`:

```
pip install -e ./tools --user
```

Now these packages can be used to run CGMF, analyze the results, and perform UQ! To see how this works, check out the files in `examples/`

# to update the dependencies:
```
git submodule update --remote --recursive
```
