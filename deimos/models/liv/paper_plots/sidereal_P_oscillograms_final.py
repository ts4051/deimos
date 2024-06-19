'''
Comparison of oscillation probability skymaps for IceCube and ARCA in RA,DEC. 
- Sidereal SME parameters and matter effects can be activated
- Earth layer boundaries are shown
- Using nuSQuIDS for oscillation calculations
Edited by Johann Ioannou-Nikolaides based on a script by Simon Hilding-Nørkjær
'''


import numpy as np
import time as time_module
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.utils.plotting import plt, dump_figures_to_pdf

#
# Main 
#
if __name__ == "__main__":
    #
    # Define basic system parameters
    #
    initial_flavor = 1  # 1 corresponds to numu
    nubar = False  # False for neutrino, True for antineutrino
    E_GeV = np.array([12.56, 20000.0])  # Energy range in GeV
    E_node = 0 # Energy node to plot
    ra_dec_grid = [10, 10]  # Grid resolution for RA/DEC

    #
    # Solver settings
    #
    solver = "nusquids"
    kw = {}
    if solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # Initialize oscillation calculators for IceCube and off-axis detectors
    IC_calculator = OscCalculator(tool=solver, atmospheric=True, **kw)
    Off_axis_calculator = OscCalculator(tool=solver, atmospheric=True, **kw)

    # Set matter effects and detectors
    IC_calculator.set_matter("earth") # "earth" or "vacuum"
    IC_calculator.set_detector("icecube")
    Off_axis_calculator.set_matter("earth")
    Off_axis_calculator.set_detector("arca")

    #
    # Set SME parameters
    #
    # Define SME parameters
    sme_basis = "mass" # "mass" or "flavor"
    flavor_structure = np.array([0.0, 0.0, 1.0]) # diagonal elements of the SME matrix
    a_eV_direction_structure = np.array([0.0, 0.0, 1.0, 0.0]) # t, x, y, z
    c_direction_structure = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]) # 2D structure 
    a_magnitude_eV = 0
    c_magnitude = 0
    a_eV = np.array([a_magnitude_eV * n * np.diag(flavor_structure) for n in a_eV_direction_structure])
    ct = np.array([c_magnitude * n * np.diag(flavor_structure) for n in c_direction_structure])
    
    ### END OF SETUP ###

    #
    # Definition of Earth layer boundaries
    #
    # Define RA/DEC grid and calculate horizons
    ra_values_deg = np.linspace(0.0, 360.0, num=ra_dec_grid[0])
    dec_values_deg = np.linspace(-90.0, 90.0, num=ra_dec_grid[1])
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    time = "July 16, 1999, 10:00"

    azimuth = np.linspace(0, 360, 1000)
    IC_RA_horizon, IC_DEC_horizon = IC_calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)
    Off_axis_RA_horizon, Off_axis_DEC_horizon = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)

    # Order off-axis horizon arrays by RA
    Off_axis_indices = np.argsort(Off_axis_RA_horizon)
    Off_axis_RA_horizon = Off_axis_RA_horizon[Off_axis_indices]
    Off_axis_DEC_horizon = Off_axis_DEC_horizon[Off_axis_indices]

    # Define Earth's layer boundaries based on the PREM model
    inner_core_radius = 1221.5
    outer_core_radius = 3480.0
    mantle_radius = 5701.0
    earth_radius_km = 6371.0

    # Earth core boundaries
    pathlength_inner_core = 2 * np.sqrt(earth_radius_km**2 - inner_core_radius**2)
    cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
    IC_RA_core, IC_DEC_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)
    Off_axis_RA_core, Off_axis_DEC_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)

    pathlength_outer_core = 2 * np.sqrt(earth_radius_km**2 - outer_core_radius**2)
    cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
    IC_RA_outer_core, IC_DEC_outer_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)
    Off_axis_RA_outer_core, Off_axis_DEC_outer_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)

    pathlength_mantle = 2 * np.sqrt(earth_radius_km**2 - mantle_radius**2)
    cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
    IC_RA_mantle, IC_DEC_mantle = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)
    Off_axis_RA_mantle, Off_axis_DEC_mantle = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)

    Off_axis_indices = np.argsort(Off_axis_RA_mantle)
    Off_axis_RA_mantle = Off_axis_RA_mantle[Off_axis_indices]
    Off_axis_DEC_mantle = Off_axis_DEC_mantle[Off_axis_indices]

    #
    # Calculate oscillation probabilities
    #
    # Initialize arrays for storing probabilities and coszen values
    P_shape = (3, len(dec_values_deg), len(ra_values_deg))
    P_IC, P_Off_axis = np.zeros(P_shape), np.zeros(P_shape)

    t_init = time_module.time()

    # Loop over DEC and RA to calculate oscillation probabilities
    for i, dec_rad in enumerate(dec_values_rad):
        for j, ra_rad in enumerate(ra_values_rad):
            if i == 0 and j == 0:
                start_time = time_module.time()
                t = 0
            t += 1

            print(f"Progress: {100. * (i * len(ra_values_rad) + j) / (len(dec_values_rad) * len(ra_values_rad)):.2f}%", end="\r")

            calc_kw = {
                "initial_flavor": initial_flavor,
                "nubar": nubar,
                "energy_GeV": E_GeV,
                "ra_rad": ra_rad,
                "dec_rad": dec_rad,
                "time": time,
            }

            P_IC_results, _, _ = IC_calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, c=ct, **calc_kw)
            P_Off_axis_results, _, _ = Off_axis_calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, c=ct, **calc_kw)

            P_IC[:, i, j] = np.squeeze(P_IC_results)[E_node]
            P_Off_axis[:, i, j] = np.squeeze(P_Off_axis_results)[E_node]

            assert np.isclose(np.sum(P_IC[:, i, j]), 1.0, atol=1e-10)
            assert np.isclose(np.sum(P_Off_axis[:, i, j]), 1.0, atol=1e-10)

            if t == 100:
                delta_time = (time_module.time() - start_time) / 100.0
                print(f"Calculation time for one iteration: {delta_time:.4f} seconds")
                print(f"Total calculation time estimate: {delta_time * len(dec_values_deg) * len(ra_values_deg) / 60.0:.2f} minutes")

    print(f"Total calculation time: {(time_module.time() - t_init) / 60.0:.2f} minutes")

    #
    # Plotting
    #
    linewidth = 2
    alpha = 1

    # Create the figure and axis objects
    fig, ax = plt.subplots(3, 2, figsize=(9, 10), sharex=True, sharey=True)
    fig.subplots_adjust(right=0.88, wspace=0.045, hspace=0.05)

    # Flatten the axis array for easy iteration
    ax = ax.flatten()

    # Title for the Plots
    fig.suptitle(f"$E$ = {E_GeV[E_node]:.3g} GeV // time: {time} // SME: a_eV_y={a_magnitude_eV:.3g} // c={c_magnitude:.3g}", fontsize=12)

    # Plot probability matrices for IceCube and Off-axis detectors
    probabilities = [P_IC, P_Off_axis]
    detectors = ["IceCube", "ARCA"]
    colors = ["RdPu"] * 6
    cbar_positions = [(0.90, 0.632, 0.02, 0.247), (0.90, 0.37, 0.02, 0.247), (0.90, 0.108, 0.02, 0.247)]
    horizons_RA = [IC_RA_horizon, Off_axis_RA_horizon, IC_RA_core, Off_axis_RA_core, IC_RA_outer_core, Off_axis_RA_outer_core, IC_RA_mantle, Off_axis_RA_mantle]
    horizons_DEC = [IC_DEC_horizon, Off_axis_DEC_horizon, IC_DEC_core, Off_axis_DEC_core, IC_DEC_outer_core, Off_axis_DEC_outer_core, IC_DEC_mantle, Off_axis_DEC_mantle]

    for i in range(3):
        for j in range(2):
            idx = 2 * i + j
            ax[idx].imshow(probabilities[j][i, :, :], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=colors[idx], vmin=0., vmax=1.)

    # Colorbars for the plots
    for i, (pos, prob_matrix) in enumerate(zip(cbar_positions, probabilities[0])):
        cbar_ax = fig.add_axes(pos)
        cbar = fig.colorbar(ax[i * 2 + 1].images[0], cax=cbar_ax, orientation="vertical", fraction=0.05, pad=0.05)
        cbar.set_label(f"${Off_axis_calculator.get_transition_prob_tex(initial_flavor, i, nubar)}$", fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.set_yticklabels(["0\n", "0.2", "0.4", "0.6", "0.8", "\n1.0"])

    # Plot overlays for horizon and core boundaries
    for i in range(3):
        for j in range(2):
            idx = 2 * i + j
            ax[idx].plot(horizons_RA[j], horizons_DEC[j], color="lime", alpha=alpha, lw=linewidth)
            ax[idx].plot(horizons_RA[j + 2], horizons_DEC[j + 2], color="red", alpha=alpha, lw=linewidth)
            ax[idx].plot(horizons_RA[j + 4], horizons_DEC[j + 4], color="orange", alpha=alpha, lw=linewidth)
            ax[idx].plot(horizons_RA[j + 6], horizons_DEC[j + 6], color="yellow", alpha=alpha, lw=linewidth)
            
            ax[idx].plot(90, 0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=7, linestyle="None")

    # Axis labels and ticks
    for i in range(3):
        for j in range(2):
            idx = 2 * i + j
            ax[idx].set_xticks([0, 90, 180, 270, 360])
            ax[idx].set_yticks([-90, -45, 0, 45, 90])
            ax[idx].set_yticklabels(["-90\n", "-45", "0", "45", "\n90"])
            ax[idx].tick_params(labelsize=14)
            if j == 0:
                ax[idx].text(0.04, 0.12, "IceCube", transform=ax[idx].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax[idx].set_ylabel("DEC[deg]", fontsize=14)
            else:
                ax[idx].text(0.04, 0.95, "ARCA", transform=ax[idx].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax[4].set_xlabel("RA[deg]", fontsize=14)
    ax[5].set_xlabel("RA[deg]", fontsize=14)
    ax[4].set_xticklabels([" 0", "90", "180", "270", "360     "])   
    ax[5].set_xticklabels([" 0", "90", "180", "270", "360     "])

    # Legend
    ax[0].plot([], [], color="lime", alpha=alpha, lw=linewidth, label="Horizon")
    ax[0].plot([], [], color="yellow", alpha=alpha, lw=linewidth, label="Mantle")
    ax[0].plot([], [], color="orange", alpha=alpha, lw=linewidth, linestyle=None, label="Outer core")
    ax[0].plot([], [], color="red", alpha=alpha, lw=linewidth, linestyle=None, label="Inner core")
    fig.legend(loc="upper center", fontsize=12, ncol=5, bbox_to_anchor=(0.5, 0.93))

    # Save the figure
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )

    # Done