# PyCobyla

A pure Python implementation of COBYLA (Constrained Optimization by
Linear Approximation). This implementation is based in COBYLA C version
ported by Jean-Sebastien Roy (js@jeannot.org) and very close to
the original implementation designed in Fortran by Michael J. D. Powell.

## Purpose

As a result of my Bachelor's Degree Final Project, one of the
objectives is understood COBYLA algorithm in deep: how it works and try to
explain to the others in a clear way.
Current implementations are fully functional but far from programming bests
practices. So, with the aim to achieve a clean implementation in
Python, here is my contribution.

## Steps

These steps are written in order to guide my Final Project Advisor but
could be useful for others.

### Install Anaconda distribution and create an conda environment

[Anaconda] (https://www.anaconda.com/products/individual)

After a clean installation, create a clean environment
```sh
%> conda create -n my-tfg python=3.8
%> conda activate my-tfg
```

### Clone the repository
```sh
%> mkdir -p jmsaxi-tfg/repos
%> cd jmsaxi-tfg/repos
%> git clone https://github.com/josepsanz/nlopt.git tfg-nlopt
%> cd tfg-nlopt
%> git checkout ft-cobyla-python
```

### Install PyCobyla as development module
At the root of the repo
```sh
%> cd src/algs/pycobyla
%> pip install -e .
```

### Run original suite test
As a part of the process to validate the correctness of this Python
port, C tests are ported too. You can run the complete suite tests.
```sh
%> pip install -r tests/requirements.txt
%> pytest tests
```

#### Run original suite test with coverage 
```sh
%> overage run --rcfile=coverage.cfg -m pytest tests
%> coverage report
%> coverage html
%> firefox htmlcov/index.html
```
