# omp-uq

Uncertainty quantification of fission observables calculated with a fission event generator with Monte Carlo Hauser-Fresbach fragment de-excitation, due to model paramater uncertainty in the [Koning-Delaroche global optical model](https://www.sciencedirect.com/science/article/pii/S0375947402013210?casa_token=ADeCX1nO83AAAAAA:Xwa6DlMKYvVU0ygGxoD0C6bfFlG0UB9hrOHojDbv2dQ7zsZvd7hhlZzvDo1b1sVxOYzL90kj) for fragment neutron emission.

This code takes in a covariance matrix or distribution of global optical model parameters, and performs UQ on a variety of fission observables and fragment-dexciation variables using a modified version of LANL's [CGMF](https://github.com/lanl/cgmf) fission event generator. The modified version used in this code lives [here](https://github.com/beykyle/cgmf), and is included as a git submodule.

The meat of this code is the python scripts in [analysis/tools](https://github.com/beykyle/omp-uq/tree/main/analysis) that run CGMF and analyze histories.

# download 

```
git clone --recurse-submodules git@github.com:beykyle/omp-uq.git
```

# install

First, [build and install](https://cgmf.readthedocs.io/en/latest/start.html#installing-cgmf) the linked submodule verison of CMGF, for example:

```
cd CGMF
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/  -Dcgmf.x.MPI=ON ..
make
sudo make install
pip install -e ./tools/ --user
```

Next, download and extract the [ENSDF files](https://www.nndc.bnl.gov/ensdfarchivals/) to a path pointed to by `$XDG_DATA_HOME/ensdf`, as described in the [Nudel documentation](https://github.com/op3/nudel#ensdf). Then, install:

```
pip install -e ./nudel --user
```

# update CGMF
```
git submodule update --remote --recursive
```
