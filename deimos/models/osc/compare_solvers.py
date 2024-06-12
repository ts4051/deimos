'''
Compare different oscillation solvers (wrapped within the OscCalculator class) to see if they agree
Tom Stuttard
'''

import sys, os, collections, datetime

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *



def compare_osc_solvers(solver_defs) :

    #TODO matter, atmospheric, nubar...


    #
    # Loop over solvers
    #

    # Init figures
    fig_nova, ax_nova = None, None
    fig_dc, ax_dc = None, None
    fig_dc2, ax_dc2 = None, None
    fig_db, ax_db = None, None
    fig_kl, ax_kl = None, None

    # Profiling
    time_taken = collections.OrderedDict()

    # Loop over solvers
    for label, solver_opts in solver_defs.items() :

        print("\n\n>>> %s..." % label)

        start_time = datetime.datetime.now()

        # Extract properties
        solver = solver_opts["solver"]
        color = solver_opts["color"]
        linestyle = solver_opts["linestyle"]
        ode_solver = solver_opts.get("ode_solver", None)


        #
        # Create model
        #

        # Tool-specific options
        kw = {}
        if solver == "deimos" :
            kw["ode_solver"] = ode_solver
        elif solver == "nusquids" :
            kw["energy_nodes_GeV"] = np.geomspace(1e-3, 1e3, num=1000) # Very wide range, covering reactor -> atmo (if have issues, might have to set nodes per experiment)

        # Create calculator
        calculator = OscCalculator(
            solver=solver,
            atmospheric=False,
            **kw
        )

        # Use vacuum
        calculator.set_matter("vacuum")



        #
        # Plot NOvA
        #

        print("\nPlot NOvA...")

        fig_nova, ax_nova, _ = calculator.plot_osc_prob_vs_energy(
            initial_flavor=1, 
            final_flavor=1, 
            nubar=False,
            energy_GeV=np.linspace(0.5, 10., num=500), # Does not like E=0
            distance_km=NOvA_BASELINE_km, 
            lw=3,
            color=color, 
            linestyle=linestyle,
            label=label,
            title="NOvA",
            fig=fig_nova,
            ax=ax_nova,
        )


        #
        # Plot DeepCore
        #

        print("\nPlot DeepCore...")

        fig_dc, ax_dc, _ = calculator.plot_osc_prob_vs_energy(
            initial_flavor=1, 
            final_flavor=2, 
            nubar=False,
            energy_GeV=np.geomspace(1., 200., num=500), 
            distance_km=EARTH_DIAMETER_km, # coszen = -1 
            lw=3,
            color=color, 
            label=label,
            linestyle=linestyle,
            title="DeepCore",
            xscale="log",
            fig=fig_dc,
            ax=ax_dc,
        )

        fig_dc2, ax_dc2, _ = calculator.plot_osc_prob_vs_distance(
            initial_flavor=1, 
            final_flavor=2, 
            nubar=False,
            energy_GeV=25., # Primary oscillation maximum for up-going numu
            distance_km=np.linspace(0., EARTH_DIAMETER_km),
            lw=3,
            color=color, 
            label=label,
            linestyle=linestyle,
            title="DeepCore",
            fig=fig_dc2,
            ax=ax_dc2,
        )


        #
        # Plot Daya Bay
        #

        print("\nPlot Daya Bay...")

        fig_kl, ax_kl, _ = calculator.plot_osc_prob_vs_energy(
            initial_flavor=0, 
            final_flavor=0, 
            nubar=True,
            energy_GeV=np.linspace(1e-3, 10.e-3, num=500), 
            distance_km=1.7, # Furthest detector
            lw=3,
            color=color, 
            label=label,
            linestyle=linestyle,
            title="Daya Bay",
            fig=fig_kl,
            ax=ax_kl,
            ylim=[0.85, 1.],
        )


        #
        # Plot KamLAND
        #

        if False : # This is slow for DEIMOS currentl;y due to fast oscillations

            print("\nPlot KamLAND...")

            fig_db, ax_db, _ = calculator.plot_osc_prob_vs_energy(
                initial_flavor=0, 
                final_flavor=0, 
                nubar=True,
                energy_GeV=np.linspace(1e-3, 10.e-3, num=1000), 
                distance_km=KAMLAND_BASELINE_km, # Furthest detector
                lw=3,
                color=color, 
                label=label,
                linestyle=linestyle,
                title="KamLAND",
                fig=fig_db,
                ax=ax_db,
            )


        # Time logging
        time_taken[label] = datetime.datetime.now() - start_time

    # Report time profiling
    print("Time taken : ")
    for k, v in time_taken.items() :
        print("%s : %s" % (k, v))


#
# Main
#

if __name__ == "__main__" :

    #
    # Compare different oscillations codes
    #

    solver_defs = collections.OrderedDict()

    solver_defs["deimos"] = {
        "solver" : "deimos",
        "color" : "blue",
        "linestyle" : "-",
    }

    solver_defs["nusquids"] = {
        "solver" : "nusquids",
        "color" : "red",
        "linestyle" : "--",
    }

    # solver_defs["prob3"] = {
    #     "solver" : "prob3",
    #     "color" : "seagreen",
    #     "linestyle" : ":",
    # }

    compare_osc_solvers(solver_defs)


    #
    # Compare different ODE solvers within DEIMOS
    #

    if False : # Can enable this if desired, but slow

        solver_defs = collections.OrderedDict()

        solver_defs["deimos (odeintw)"] = {
            "solver" : "deimos",
            "ode_solver" : "odeintw",
            "color" : "blue",
            "linestyle" : "-",
        }

        solver_defs["deimos (solve_ivp)"] = {
            "solver" : "deimos",
            "ode_solver" : "solve_ivp",
            "color" : "red",
            "linestyle" : "--",
        }

        compare_osc_solvers(solver_defs)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
