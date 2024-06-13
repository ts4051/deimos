'''
Comparing a range of QG-motivated decoherence scenarios

This is part of addressing comments on the review of the IceCube TeV decoherence paper

Refs:
  [1] https://arxiv.org/pdf/2306.14778.pdf
  [2] https://arxiv.org/abs/2208.12062

Tom Stuttard
'''

import sys, os, collections, copy, datetime

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

from deimos.density_matrix_osc_solver.density_matrix_osc_solver import GeV_to_eV


#
# Helper functions
#

def calc_qg_damping_coefficient(qg_model, E_GeV, mass_splitting_eV2, theta_rad, **qg_params) :
    '''
    Calculate the damping coefficient for the various models considered, alpha

    Definition : D = alpha * L
    Damping term: exp{-D}

    This is only valid for the 2-flavor vacuum case
    '''

    # Convert units
    E_eV = E_GeV * GeV_to_eV

    # Get QG scale
    E_qg_eV = qg_params["E_qg_eV"]

    # Checks
    assert E_qg_eV is not None
    assert qg_model is not None
    assert isinstance(qg_model, str)

    # Some elements are common to multiple models
    if qg_model in ["minimal_length_fluctuations", "metric_fluctuations"] :

        # Mass of lightest neutrino state is a free parameter
        m1_eV = qg_params["m1_eV"]
        m2_eV = np.sqrt(  mass_splitting_eV2 + np.square(m1_eV) ) #TODO think I might have made a function for this already? if so use it, it not then make one

        # Calc average group velocity, p/E, where p is average of the two states
        # Using E^2 = p^2 + m^2 to get p, where again m is the average
        # Note that this is basically always = 1 for the relevent energies (as also mentioned in footnote of page 12 of https://arxiv.org/pdf/2306.14778.pdf)
        #TODO cross-check calc via Lorentz boost method?
        m_mean_eV = ( m1_eV + m2_eV ) / 2.
        p_mean_eV = np.sqrt( np.square(E_eV) - np.square(m_mean_eV) )
        v_g = p_mean_eV / E_eV
        
        # Calc (delta m)^2   --> NOT delta (m^2)   (e.g. NOT mass splitting)
        dm_squared_eV = np.square( m2_eV - m1_eV )  #TODO try alt calc
    
    # Calc damping term: Minimal length fluctuations case
    if qg_model == "minimal_length_fluctuations" :
        alpha = 16. * np.power(E_eV, 4.) * dm_squared_eV
        alpha /= (v_g * np.power(E_qg_eV, 5.) )

    # Calc damping term: Stochastic fluctuations of the metric case
    #TODO eqn 29 vs 30? think 30 is just a simplification of 29 in some cases. For now using eqn 29
    elif qg_model == "metric_fluctuations" :
        alpha = 1. / ( 8. * v_g * E_qg_eV )
        alpha *= np.square( 1. + ( np.square(E_eV) / np.square(m1_eV*m2_eV) ) ) 
        alpha *= dm_squared_eV

    # Calc damping term: nu-VBH interactions
    elif qg_model == "nu_vbh_interactions" : #TODO which interaction type?
        # https://arxiv.org/pdf/2007.00068.pdf eqn 20, with zeta = 1 for natural scale (and using E_qg as free param as with other models here, e.g. not necessarily equal to M_P)
        n = qg_params["n"]
        alpha = np.power(E_eV, n) / np.power(E_qg_eV, n-1.)

    else :
        raise Exception("Unknown model : %s" % qg_model)

    return alpha


def calc_coherence_length(*args, **kwargs) :

    '''
    Get the coherence length for a given QG model

    This is when exp{-alpah*L} = 1/e   ->  -alpha*L = -1   -> L_coh = 1/alpha
    '''
    return 1. / calc_qg_damping_coefficient(*args, **kwargs)


    
#
# Plot functions
#

def compare_qg_models(solver) :
    '''
    Compare decoherence resulting from a range on QG models, for the case of atmospheric neutrinos
    '''

    pass #TOOD plot osc probs for each model


def compare_qg_models_coherence_length() :
    '''
    Compare the natural coherence length for a range of QG models
    '''

    # Steer standard osc physics
    mass_splitting_eV2 = MASS_SPLITTINGS_eV2[-1] # atmo
    mixing_angle_rad = MIXING_ANGLES_rad[-1] # atmo

    # Steer QG
    E_qg_eV = PLANCK_MASS_eV
    m1_eV = 1.

    # Decide E range
    E_eV = np.logspace(0., 30, num=100)

    # Make figure
    fig, ax = plt.subplots( figsize=(6, 4) )

    # Mark Planck scale
    ax.axvline(PLANCK_MASS_eV, linestyle="-", lw=1, color="black", label="Planck scale")
    ax.axhline(PLANCK_LENGTH_m, linestyle="-", lw=1, color="black")

    # Mark Earth diameter
    ax.axhline(EARTH_DIAMETER_km*1e3, linestyle="-", lw=1, color="brown", label="Earth diameter")

    # Calc and plot coherence length for each model
    L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="minimal_length_fluctuations", E_qg_eV=E_qg_eV, m1_eV=m1_eV) * 1e3 # km -> m
    ax.plot(E_eV, L_coh_m, color="red", label="Minimal length fluctuations", linestyle="-", lw=2)

    L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="metric_fluctuations", E_qg_eV=E_qg_eV, m1_eV=m1_eV) * 1e3 # km -> m
    ax.plot(E_eV, L_coh_m, color="blue", label="Metric fluctuations", linestyle="-", lw=2)

    for n, linestyle in zip([0, 1, 2, 3], ["-", "--", "-.", ":"]) :
        L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="nu_vbh_interactions", E_qg_eV=E_qg_eV, n=n) * 1e3 # km -> m
        ax.plot(E_eV, L_coh_m, color="orange", label=r"$\nu$-VBH ($n$=%i)"%n, linestyle=linestyle, lw=2)

    # Format
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\nu}$ [eV]")
    ax.set_ylabel(r"$L_{coh}$ [m]")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()




#
# Main
#

if __name__ == "__main__" :

    solver = "deimos"

    compare_qg_models_coherence_length()
    compare_qg_models(solver=solver)

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
