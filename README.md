# omp-uq
Uncertainty quantification of Koning-Delaroche global optical model parameters in fission fragment de-excitation

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
