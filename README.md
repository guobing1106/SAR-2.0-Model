# SAR 2.0 Box 5 Simulations

This repository contains the simulation scripts used for **Box 5** in the manuscript:

**Reframing the Species–Area Relationship: A Process-Based and Time-Explicit SAR 2.0 Framework**

These scripts were developed to illustrate the core idea of **SAR 2.0**. They are intended as **proof-of-concept simulations** of model behavior under asynchronous dynamics, rather than empirically parameterized forecasts for any particular ecosystem.

## What this repository contains

The scripts in this repository correspond to the four parts of **Box 5**:

1. **Unification of SAR 2.0 with classical theory**  
   Parameter-space exploration showing that under some conditions, the SAR 2.0 framework can approximate a stable classical power-law SAR.

2. **Simulation of asynchronous dynamics among the three processes**  
   Simulations showing that asynchronous fluctuations in speciation, dispersal, and extinction can generate positive, negative, and non-significant SARs through time.

3. **Reproduction of temporal discontinuity under extinction dominance**  
   Simulations showing that extinction-driven dynamics alone can generate temporal discontinuity in SAR patterns.

4. **Regulatory effects of fluctuation amplitude or baseline intensity**  
   Sensitivity analyses showing how changes in the relative strengths of the three processes affect the proportions of SAR states.

## Conceptual background

In the SAR 2.0 framework, the species–area relationship is treated as a dynamic outcome jointly shaped by three processes:

- speciation
- dispersal
- extinction

The scripts implement illustrative functions for these three processes and explore how their time-varying and asynchronous dynamics may reshape SAR patterns.

## Script overview

The scripts are numbered in the recommended execution order. All output files (images, CSV, PDF, SVG) are saved to a `results/` folder created automatically in the directory where the script is run.

| File name | Corresponding Box 5 part | Description |
|-----------|--------------------------|-------------|
| `01_z_value_convergence.py` | 1 | Scans the extended parameter space to identify conditions under which the time‑varying exponent `z` converges towards a stable, classical power‑law SAR. Generates parameter‑importance plots, correlation matrices, and distribution histograms. |
| `02_three_process_dynamics.py` | 2 | Visualizes the temporal dynamics of diffusion, extinction, speciation, and the net `z`‑value (`D – E + Sp`). Produces four separate figures for the three processes and the net fluctuation. |
| `03_sar_type_proportions.py` | 2 | Simulates the proportions of positive, negative, and non‑significant SAR patterns through time under a representative asynchronous parameterisation. Includes strict outlier removal, boxplots, and a pie‑chart summary. |
| `04_extinction_dominance.py` | 3 | Extinction‑dominated simulation that examines temporal discontinuity and the alternation of SAR states. Produces boxplots, a pie chart, and multiple SAR power‑law visualisations (log‑log, linear, temporal dynamics, and a single representative fit). |
| `05_baseline_effects.py` | 4 | Sensitivity analysis for extinction‑dominated scenarios. Generates a forest‑plot style visualisation of SAR state transitions through time, with positive, negative, and neutral correlations colour‑coded. |
| `06_forest_plot_and_summary.py` | 4 | Sensitivity analysis showing how changing the baseline values of diffusion, extinction, or speciation affects the proportions of SAR types. Produces three independent figures (diffusion, extinction, speciation impacts), each saved as PNG, PDF and SVG. |

## Requirements

These scripts were developed in Python 3 and use the following main packages:

- numpy
- matplotlib
- pandas
- scipy
- tqdm
- scikit-learn

You can install them with:

```bash
pip install -r requirements.txt