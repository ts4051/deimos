# Standard Model Extension implementation for the study of Lorentz and CPT Violation in neutrino oscillations 
## Overview
This repository contains implementations of Lorentz and CPT violation in two neutrino oscillation solvers:
- **DEIMOS**: A simple Python-based neutrino oscillation solver using the density matrix formalism.
- **nuSQuIDS**: A neutrino oscillation software using the SQuIDS framework.
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
## Introduction
The study of Lorentz and CPT violation in neutrino oscillations provides insights into fundamental symmetries of nature. This project extends the capabilities of DEIMOS and nuSQuIDS to incorporate these effects based on the Standard Model Extension [arXiv:1112.6395](https://arxiv.org/abs/1112.6395), allowing for detailed analyses and simulations. 
## Features
- **DEIMOS**:
  - Minimal SME (all mass-independent, renormalizable operators inducing (anti) &nu; &rarr; &nu; oscillations)
  - Renormalizable SME - **coming soon**
  - Easy-to-use and flexible for various neutrino oscillation scenarios
  - Supports vacuum and constant density matter potentials
- **nuSQuIDS**:
  - Minimal SME (all mass-independent, renormalizable operators inducing (anti) &nu; &rarr; &nu; oscillations)
  - Renormalizable SME - **coming soon**
  - Handles complex neutrino oscillation problems in different environments including full PREM model DOI:[10.1016/0031-9201(81)90046-7](https://www.sciencedirect.com/science/article/pii/0031920181900467)
## Installation
### Prerequisites
- Same as for [DEIMOS](https://github.com/ts4051/deimos)
- [nuSQuIDS dependencies](https://github.com/arguelles/nuSQuIDS#installation)
### Installing nuSQuIDS
- Use nuSQuIDS repository `git@github.com:ts4051/nuSQuIDS` and `bsm` branch 
## Usage
- Specify solver (**deimos** or **nusquids**) when setting calculator `OscCalculator(tool=solver, **kw)`. 
- (Optional) Set matter `vacuum` or `constant` for **deimos** and **nusquids** or `earth` or `layers` for **nusquids**. 
- The `calc_osc_prob_sme` method calculates neutrino oscillation probabilities taking sidereal SME effects, which depend on righ ascension, declination, and time into account. 
    - Input: 
        - *energy_GeV* (float or array): Neutrino energy in GeV.
        - *ra_rad* (float or array): Right ascension in radians.
        - *dec_rad* (float or array): Declination in radians.
        - _time_ (str or array): Time in a format parsable by astropy.time.Time.
        - *initial_flavor* (int): Initial neutrino flavor (0: electron, 1: muon, 2: tau).
        - *nubar* (bool, optional): Indicates if antineutrino (default: False).
        - *std_osc* (bool, optional): Toggle standard oscillations without SME (default: False).
        - *basis* (str, optional): Basis for SME calculations ("flavor", "mass", etc.).
        - *a_eV* (array, optional): SME parameter a in eV.
        - *c* (array, optional): SME parameter c.
    - Returns:
        - *osc_probs* (array): Oscillation probabilities.
        - *coszen_values* (array): Cosine of the zenith angles (if atmospheric).
        - *azimuth_values* (array): Azimuth angles (if atmospheric).
- The `set_sme` method configures the Standard Model Extension (SME) parameters for neutrino oscillation calculations. It supports both sidereal and isotropic SME models. It can be used in combination with other methods of `osc_calculator.py` such as `calc_osc_prob`
    - Parameters:
        - *directional* (bool): Indicates if the SME model is directional (sidereal).
        - *basis* (str, optional): The basis for SME calculations, either "mass" or "flavor" (default: "mass").
        - *a_eV* (numpy.ndarray, optional): SME parameter a in eV with shape (3, $N_\nu$, $N_\nu$) for directional or ($N_\nu$, $N_\nu$) for isotropic.
        - *c* (numpy.ndarray, optional): SME parameter c with the same shape requirements as a_eV.
        - *ra_rad* (float, optional): Right ascension in radians (required for directional SME).
        - *dec_rad* (float, optional): Declination in radians (required for directional SME).
## Examples
- See [sidereal_P_oscillograms_final.py](https://github.com/ts4051/deimos/tree/main/deimos/models/liv/paper_plots/sidereal_P_oscillograms_final.py) and [sidereal_time_dependence_final.py](https://github.com/ts4051/deimos/tree/main/deimos/models/liv/paper_plots/sidereal_time_dependence_final.py) and [plot_sme.py](https://github.com/ts4051/deimos/tree/main/deimos/models/liv/plot_sme.py) for example implementations of the SME in neutrino oscillations.