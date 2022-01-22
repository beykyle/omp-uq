# omp-uq

Uncertainty quantification of fission and fission fragment de-excitation observables, due to model paramater uncertainty in the [Koning-Delaroche global optical model](https://www.sciencedirect.com/science/article/pii/S0375947402013210?casa_token=ADeCX1nO83AAAAAA:Xwa6DlMKYvVU0ygGxoD0C6bfFlG0UB9hrOHojDbv2dQ7zsZvd7hhlZzvDo1b1sVxOYzL90kj) for fragment neutron emission.

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
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/  ..
make
sudo make install
```
