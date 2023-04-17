'''
A set of useful constants

Tom Stuttard
'''

import numpy as np
from analysis.common.utils.natural_units import si_to_natural_units, natural_to_si_units

#
# Fundamental physics
#

# Number of neutrino (flavor/mass) states
NUM_STATES = 3

# Neutrino bases
BASES = ["mass","flavor"]

# Some useful distances
EARTH_DIAMETER_km = 12742.
EARTH_SUN_DISTANCE_m = 148.92e9
MILKY_WAY_DIAMETER_m = 1e21
EARTH_TXS_RED_SHIFT = 0.3365 # From arXiv 1802.01939
EARTH_TXS_DISTANCE_m = 3.66e25 # Computed using astropy.cosmology.Planck15.lookbacktime(EARTH_TXS_RED_SHIFT) * c #TODO tension with wikipedia number, look into
OBSERVABLE_UNIVERSE_DIAMETER_m = 8.8e26 # From google, this is 93 billion lightyears
MOST_DISTANT_GRB_RED_SHIFT = 9.4 # GRB 090429B, z ~ 9.4 (https://arxiv.org/pdf/1105.4915.pdf) #TODO Is this a short or long GRB?

# Define the neutrino mass matrix
# Expressing as the delta mass (mass splittings) matrix (e.g. relative to m1, assuming normal ordering)
MASS_SPLITTINGS_eV2 = np.array([ 7.39e-5, 2.528e-3 ]) # [21, 31] nufit v4.1, NO, with SK atmo data

# Mixing angles
MIXING_ANGLES_rad = np.deg2rad( np.array([ 33.82, 8.60, 48.6 ]) ) # nufit v4.1, NO, with SK atmo data

# CP violating phase
DELTACP_rad =  np.deg2rad(221.) # nufit v4.1, NO, with SK atmo data

# Planck scale
PLANCK_MASS_eV = 1.22e28
PLANCK_LENGTH_m = 1.62e-35 #TODO 1/M_Planck
PLANCK_TIME_s = 5.39e-44 #TODO calc from M_Planck


#
# Labels
#

MASS_TEX = [r"%i"%i for i in [1,2,3]]
FLAVORS_TEX = [r"e",r"\mu",r"\tau"]
NU_MASS_TEX = [r"\nu_{%s}"%m for m in MASS_TEX]
NU_FLAVORS_TEX = [r"\nu_{%s}"%f for f in FLAVORS_TEX]

NU_COLORS = ["blue","red","green"]

ENERGY_LABEL = r"$E$ [GeV]"
DISTANCE_LABEL = r"$L$ [km]"
ZENITH_LABEL = r"$\theta_{\rm{zenith}}$"
COSZEN_LABEL = r"$\cos(\theta_{\rm{zenith}})$"
EARTH_DIAMETER_TEX = r"L_{\oplus}"
EARTH_SUN_DISTANCE_TEX = r"L_{\odot}"
TXS_LABEL = "TXS 0506+056"
MOST_DISTANT_GRB_LABEL = "GRB 090429B"

PLANCK_LABEL = r"\rm{Planck}"
PLANCK_MASS_TEX = r"M_{%s}" % PLANCK_LABEL
PLANCK_LENGTH_TEX = r"L_{%s}" % PLANCK_LABEL
PLANCK_TIME_TEX = r"t_{%s}" % PLANCK_LABEL


#
# Experiments
#

# Define some other experiments
NOvA_BASELINE_km = 810.
NOvA_ENERGY_GeV = 2.
MINOS_BASELINE_km = 735.
MINOS_ENERGY_GeV = 4.
DUNE_BASELINE_km = 1300.
DUNE_ENERGY_GeV = 3.
T2K_BASELINE_km = 295.
T2K_ENERGY_GeV = 0.65
# T2HK_BASELINE_km = TODO
T2HK_ENERGY_GeV = T2K_ENERGY_GeV
T2HKK_BASELINE_km = 1100.
T2HKK_ENERGY_GeV = T2HK_ENERGY_GeV
P2O_BASELINE_km = 2588.
# P2O_ENERGY_GeV = TODO


#
# Papers
#


# Define a common figure width
# This helps make consistent plots for papers
FIG_WIDTH = 7 # Works well for PRD 2 column format

# Define a reference coherence length
REF_COHERENCE_LENGTH_m = EARTH_DIAMETER_km * 1.e3

# Choose reference energy scale
REF_E0_eV = 1.e9 # GeV


#
# Helper functions
#

def get_default_neutrino_definitions(num_states) :
    '''
    Function to get default suite of mass splittings, mixing angles, etc
    If only 2 states, choose atmo sector
    '''

    assert num_states in [2,3]

    mass_splitting_eV2 = MASS_SPLITTINGS_eV2 if num_states == 3 else [MASS_SPLITTINGS_eV2[-1]]

    mixing_angles_rad = MIXING_ANGLES_rad if num_states == 3 else [MIXING_ANGLES_rad[-1]] 

    deltacp = None if num_states == 2 else DELTACP_rad

    mass_tex = MASS_TEX if num_states == 3 else MASS_TEX[1:]
    nu_mass_tex = NU_MASS_TEX if num_states == 3 else NU_MASS_TEX[1:]
    flavors_tex = FLAVORS_TEX if num_states == 3 else FLAVORS_TEX[1:]
    nu_flavors_tex = NU_FLAVORS_TEX if num_states == 3 else NU_FLAVORS_TEX[1:]
    nu_colors = NU_COLORS if num_states == 3 else NU_COLORS[1:]

    return mass_splitting_eV2, mixing_angles_rad, deltacp, mass_tex, nu_mass_tex, flavors_tex, nu_flavors_tex, nu_colors


