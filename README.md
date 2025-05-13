# Impact of Plasticity-Based Reservoir Adaptation on Spectral Radius and Performance of ESNs

This repository holds the implementation and the results corresponding to the paper *Impact of Plasticity-Based Reservoir Adaptation on Spectral Radius and Performance of ESNs*.

## Structure of the Repository

The repository is structured as follows:
- The notebook `impact_of_plasticity.ipynb` contains the implementation of the experiments described in the paper. Helper functions that are used in this notebook are implemented in `helpers.py`.
- The folder `results/` contains the results that our ESN achieved on the considered benchmarks, namely on the Mackey-Glass (`results/MG/`), the NARMA (`results/NARMA/`), and the Lorenz attractor series (`results/lorenz/`). Each of these three subfolders holds six files:
    - `result_esn.json`: results of the ESN without pretraining
    - `result_ao.json`: results after the pretraining with the *anti-Oja's rule*
    - `result_nah.json`: results after the pretraining with the *normalized anti-Hebbian rule*
    - `result_bcm.json`: results after the pretraining with the *BCM rule*
    - `result_dtbcm.json`: results after the pretraining with the *DT-BCM rule*
    - `result_IP.json`: results after the pretraining with *IP*
- The folder `analysis/` contains the results of the experiments related to the spectral radius. Each of the subfolders corresponding to the three considered benchmarks holds three files:
    - `nrmse_vs_sr.json`: average *NRMSE* and its standard deviation over 20 independent runs for spectral radius values $\rho = 0.05, 0.1, \dots, 1.1$
    - `nrmse_vs_sr.png`: plot of the data from `nrmse_vs_sr.json`
    - `sr_vs_epoch.png`: plot showing the change of the spectral radius over the epochs of the different pretraining methods (the spectral radius values that the plot is created from are stored in the result files that can be found in `results/`)

## Reproduction of the Results

If you want to reproduce the results from the paper, start by creating the conda environment `plasticity_env` containing the required packages with:

```
conda env create -f env.yml
```

Then open the notebook `impact_of_plasticity.ipynb` and select the created `plasticity_env` as kernel. Now simply go through the notebook which will guide you through all of our experiments.

## Contact

In case of any questions, please contact me at `franziska.fw.weber@fau.de`.
