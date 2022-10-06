# omp-uq

Uncertainty quantification of fission observables calculated with a fission event generator with Monte Carlo Hauser-Fresbach fragment de-excitation, due to model paramater uncertainty in the [Koning-Delaroche global optical model](https://www.sciencedirect.com/science/article/pii/S0375947402013210?casa_token=ADeCX1nO83AAAAAA:Xwa6DlMKYvVU0ygGxoD0C6bfFlG0UB9hrOHojDbv2dQ7zsZvd7hhlZzvDo1b1sVxOYzL90kj) for fragment neutron emission.

This code takes in a covariance matrix or distribution of global optical model parameters, and performs UQ on a variety of fission observables and fragment-dexciation variables using a modified version of LANL's [CGMF](https://github.com/lanl/cgmf) fission event generator. The modified version used in this code lives [here](https://github.com/beykyle/cgmf), and is included as a git submodule.

The meat of this code is the python modules in [tools/](https://github.com/beykyle/omp-uq/tree/main/tools/) that run CGMF and analyze histories.

# dependencies
Aside from the aforementioned submodules, this workflow is set up to run CGMF with MPI using [mpirun](https://www.open-mpi.org/doc/current/man1/mpirun.1.php). That means it should be compiled with an MPI implementation, as shown below. For heavy UQ, you'll definitely want to use MPI!

Additionally, a fairly standard suite of Python packages is used. A catch-all environment, like [Anaconda](anaconda.com) should cover everything.

# download 

```
git clone --recurse-submodules git@github.com:beykyle/omp-uq.git
```

# install

First, [build and install](https://cgmf.readthedocs.io/en/latest/start.html#installing-cgmf) the linked submodule verison of CMGF, for example:

```
mkdir CGMF/build
cd CGMF/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/  -Dcgmf.x.MPI=ON ..
make
make test
sudo make install
cd ../..
pip install -e ./tools/ --user
```

Next, download and extract the [ENSDF files](https://www.nndc.bnl.gov/ensdfarchivals/) to a path pointed to by `$XDG_DATA_HOME/ensdf`, as described in the [Nudel documentation](https://github.com/op3/nudel#ensdf). Then, install:

```
pip install -e ./nudel --user
```

Finally, install all the packages in `./tools`:

```
pip install -e ./tools --user
```

Now these packages can be used to run CGMF, analyze the results, and perform UQ! To see how this works, check out the files in `examples/`


# to update the dependencies:
```
git submodule update --remote --recursive
```
