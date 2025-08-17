# Mean-covariance robust risk measurement
This repo contains all source code that reproduce the experiments in our [Mean-covariance robust risk measurement](https://arxiv.org/pdf/2112.09959) paper.

**We welcome any feedback and suggestions! Note that we put in maximum effort to write high quality codes. However, they may still contain bugs or not be efficient enough.**

## Prerequisites
All optimization problems are implemented in Python. The implementations rely on the following third-party software: [Gurobi](https://www.gurobi.com/), [Mosek](https://www.mosek.com/), [Julia](https://julialang.org/) and [JuMP](https://jump.dev/JuMP.jl/stable/).

## Reproducing the results
First, clone the repo

> $ git clone https://github.com/sorooshafiee/mean-covariance-robust-risk.git

Then, install all required libraries for Python 3.13.5 using

> $ conda create -n rfolio python=3.13.5

> $ conda activate rfolio

> $ conda install --yes --file requirements.txt

To reproduce the numerical results, you need to first run `fig1_results.py`, `fig2_results.py`, `table2_results.py`, and `table3_results.py`. Then, run `fig1_plotter.py` and `fig2_plotter.py`.
