'''
Script for producing 1D plots of sidereal time dependence of parameters in the SME model in RA/Dec coordinates

Edited by Johann Ioannou-Nikolaides based on a script by Simon Hilding-Nørkjær
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import calc_path_length_from_coszen

def main():
    # Define basic system parameters
    detector = "arca"     # "arca" or "icecube"
    initial_flavor = 1    # numu 
    nubar = False         # neutrino or antineutrino
    E_GeV = np.array([1000., 20000.]) # Energy range in GeV
    E_node = 0              # Energy node to plot
    solver = "nusquids"     # "nusquids" or "deimos"

    #
    # Set SME parameters
    #
    # Define SME parameters
    sme_basis = "mass" # "mass" or "flavor"
    flavor_structure = np.array([0.0, 0.0, 1.0]) # diagonal elements of the SME matrix
    a_eV_direction_structure = np.array([0.0, 0.0, 1.0, 0.0]) # t, x, y, z
    c_direction_structure = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]) # 2D structure 
    a_magnitude_eV = 5e-14
    c_magnitude = 0
    a_eV = np.array([a_magnitude_eV * n * np.diag(flavor_structure) for n in a_eV_direction_structure])
    for n in c_direction_structure:
        ct = np.array([c_magnitude * m * np.diag(flavor_structure) for m in n]).reshape(4, 4, 3, 3)

    # Create oscillation calculators
    kw = {"energy_nodes_GeV": E_GeV, "nusquids_variant": "sme"} if solver == "nusquids" else {}
    calculatorIC = OscCalculator(tool=solver, atmospheric=True, **kw)
    calculatorARCA = OscCalculator(tool=solver, atmospheric=True, **kw)

    # Set detector and matter type
    calculatorIC.set_matter("vacuum") # "vacuum" or "earth"
    calculatorIC.set_detector("icecube")
    calculatorARCA.set_matter("vacuum")
    calculatorARCA.set_detector("arca")

    # Define time array
    hour_array = np.arange(1, 25, 2)
    time_array = [f"July 16, 1999, {time}:00" for time in hour_array]
    time_array = np.array(time_array)

    # Define coordinates
    ra_deg = 90
    dec_deg = np.linspace(-90, 90, num=100)
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    ### END OF SETUP ###

    # Initialize arrays for storing results
    cosz_IC = np.zeros(len(dec_deg))
    P_IC = np.zeros((3, len(dec_deg)))
    cosz_ARCA = np.zeros((len(dec_deg), len(time_array)))
    P_ARCA = np.zeros((3, len(dec_deg), len(time_array)))

    # Loop over declinations and time to calculate oscillation probabilities
    for dec_index, dec in enumerate(dec_deg):
        calc_kw = {
            "initial_flavor": initial_flavor,
            "nubar": nubar,
            "energy_GeV": E_GeV,
            "ra_rad": ra_rad,
            "dec_rad": dec_rad[dec_index],
        }

        for time_index, time in enumerate(time_array):
            print(f"Progress: {dec_index}/{len(dec_deg)} {time_index}/{len(time_array)}", end="\r")
            P_ARCA_value, coszen_ARCA_value, _ = calculatorARCA.calc_osc_prob_sme(
                basis=sme_basis, a_eV=a_eV, c=ct, time=time, **calc_kw)
            cosz_ARCA[dec_index, time_index] = coszen_ARCA_value
            P_ARCA[:, dec_index, time_index] = P_ARCA_value[E_node, :]

        P_IC_value, coszen_IC_value, _ = calculatorIC.calc_osc_prob_sme(
            basis=sme_basis, a_eV=a_eV, c=ct, time=time_array[0], **calc_kw)
        cosz_IC[dec_index] = coszen_IC_value
        P_IC[:, dec_index] = P_IC_value[E_node, :]

    # Convert cosz to baseline
    baseline_ARCA = calc_path_length_from_coszen(cosz_ARCA)
    baseline_IC = calc_path_length_from_coszen(cosz_IC)

    # Plot results
    colors = plt.cm.jet(hour_array / 24)
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(3, 1, figsize=(6, 14))

    for time_index, time in enumerate(time_array):
        if time_index + 1 == len(time_array): break
        ax[0].plot(dec_deg, cosz_ARCA[:, time_index + 1], color=colors[time_index])
        ax[1].plot(dec_deg, baseline_ARCA[:, time_index + 1], color=colors[time_index])
        ax[2].plot(dec_deg, P_ARCA[1, :, time_index + 1], color=colors[time_index])

    ax[0].plot(dec_deg, cosz_ARCA[:, 0], label="ARCA", color=colors[5])
    ax[1].plot(dec_deg, baseline_ARCA[:, 0], label="ARCA", color=colors[5])
    ax[2].plot(dec_deg, P_ARCA[1, :, 0], label="ARCA", color=colors[5])

    ax[0].plot(dec_deg, cosz_IC, "--k", lw=2, label="IC")
    ax[1].plot(dec_deg, baseline_IC, "--k", lw=2, label="IC")
    ax[2].plot(dec_deg, P_IC[1, :], "--k", lw=2, label="IC")

    # Customize plots
    ax[2].set(ylim=(-0.05, 1.05))
    for axis in ax:
        axis.legend()
        axis.set_xlabel("Declination [deg]", fontsize=16)
        axis.set_xticks([-90, -45, 0, 45, 90])
        axis.grid(alpha=0.2)
    ax[0].set_ylabel(r"$\cos(\theta_{Zenith})$", fontsize=16)
    ax[1].set_ylabel("Baseline [km]", fontsize=16)
    ax[2].set_ylabel(r"$P(\nu_\mu \to \nu_\mu)$", fontsize=16)

    # Add colorbars for time
    norm = mpl.colors.Normalize(vmin=0, vmax=24)
    cbar_axes = [fig.add_axes([0.92, 0.655 - 0.273 * i, 0.02, 0.226]) for i in range(3)]
    for cax in cbar_axes:
        cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.jet, norm=norm, orientation='vertical')
        cb.set_label('Time [h]')
        cb.set_ticks(np.arange(0, 25, 4))
        cb.set_ticklabels(np.arange(0, 25, 4))

    fig.suptitle(f"a = {a_magnitude_eV} eV, E = {E_GeV[E_node]} GeV, Vacuum", fontsize=16)

    # Save the figure
    plt.savefig(__file__.replace(".py", ".pdf"), bbox_inches='tight')
    print("Dumped file to", __file__.replace(".py", ".pdf"))


if __name__ == "__main__":
    main()