'''
Plot diffuse astrophysical flavor triangle, assuming standard oscillations

Tom Stuttard
'''

import sys, os, collections

import ternary # This is used for triangle plots

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import *
from deimos.density_matrix_osc_solver.density_matrix_osc_solver import get_pmns_matrix


#
# Main
#

if __name__ == "__main__" :


    #
    # Define physics
    #

    # PMNS matrix is the only thing that matters
    PMNS = get_pmns_matrix(theta=MIXING_ANGLES_rad, dcp=0.)

    flavors = [0, 1, 2]


    #
    # Define sources
    #

    source_cases = collections.OrderedDict()

    source_cases["Neutron decay"] = {
        "initial_flux" : np.array([1., 0., 0.]),
        "color" : "red",
    }

    source_cases["Pion decay"] = {
        "initial_flux" : np.array([1., 2., 0.]),
        "color" : "orange",
    }

    source_cases["Pion damped"] = {
        "initial_flux" : np.array([0., 1., 0.]),
        "color" : "dodgerblue",
    }


    #
    # Generate expectation at Earth
    #

    # Loop over sources
    for source_case_name, source_case_dict in source_cases.items() :

        # Init final flux container
        final_flux = np.zeros(len(flavors))

        # Loop over channels
        for initial_flavor in flavors :
            for final_flavor in flavors :

                # Calculation channel probability
                p_avg = oscillation_averaged_transition_probability( pmns=PMNS, initial_flavor=initial_flavor, final_flavor=final_flavor )

                # Add this channel's contribution to overall fluxx
                final_flux[final_flavor] += ( p_avg * source_case_dict["initial_flux"][initial_flavor] )

        # Normalise
        source_case_dict["final_flux"] = final_flux / source_case_dict["initial_flux"].sum()
        source_case_dict["initial_flux"] = source_case_dict["initial_flux"] / source_case_dict["initial_flux"].sum()


    #
    # Plot
    #

    # Make fig
    scale = 100
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(6, 5)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=10)

    # Set Axis labels and Title
    fontsize = 12
    offset = 0.14
    tax.left_axis_label(r"$\nu_\tau$ fraction [$\%$]", fontsize=fontsize, offset=offset)
    tax.right_axis_label(r"$\nu_\mu$ fraction [$\%$]", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label(r"$\nu_e$ fraction [$\%$]", fontsize=fontsize, offset=offset)

    # Loop over sources and plot
    for source_case_name, source_case_dict in source_cases.items() :
        tax.line(scale*source_case_dict["initial_flux"], scale*source_case_dict["final_flux"], linewidth=3., color=source_case_dict["color"], linestyle=":") # Line from initial to final flavor composition
        tax.scatter([scale*source_case_dict["final_flux"]], marker='o', color=source_case_dict["color"], linewidth=2, s=75, edgecolor="black", zorder=20, label=source_case_name ) # Marker for final flavor composition

    # Legend
    tax.legend(loc='upper right', fontsize=10)

    # Final formatting
    tax.ticks(axis='lbr', multiple=20, linewidth=1, offset=0.025)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.show() # This might complain, but is needed to make axis labels appear


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
