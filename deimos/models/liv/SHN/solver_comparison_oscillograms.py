'''
Compare nuSQUIDS and DEIMOS in oscillograms with sidereal SME parameters activated

Simon Hilding-Nørkjær
'''



import sys, os, collections, datetime
from astropy.time import Time

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import calc_path_length_from_coszen, get_coszen_from_path_length


#
# Main
#

if __name__ == "__main__" :


    #
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=1000)
    coszen = -1.


    #
    # Create model
    #
    detector_name = "dune" # IceCube DUNE
    num_neutrinos = 3
    
    # Create calculator
    deimos_calculator = OscCalculator(
        tool="deimos",
        atmospheric=False,
        num_neutrinos=num_neutrinos,)
    deimos_calculator.set_matter("vacuum")
    deimos_calculator.set_detector(detector_name)

    nusquids_calculator = OscCalculator(
        tool="nusquids",
        atmospheric=False,
        num_neutrinos=num_neutrinos,
        energy_nodes_GeV = np.geomspace(E_GeV[0], E_GeV[-1], num=100))
    nusquids_calculator.set_matter("vacuum")
    nusquids_calculator.set_detector(detector_name)



    #
    # Define basic system
    #

    initial_flavor, final_flavor, nubar = 1, 0, False # numu survival
    E_GeV = 100.


    #
    # Define physics cases
    #

    cases = collections.OrderedDict()

    directional = True
    sme_basis = "mass"

    a_magnitude_eV = 1e-13 # Overall strength of a component
    c_magnitude = 0 #1e-26 # Overall strength of c component


    flavor_structure = np.array([0., 0., 1.]) # Equally shared between flavors (diagonal)

    if directional :
        direction_structure = np.array([0., 0., 1.]) # Orient field in x direction
        cases["a"] = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in direction_structure ])
        null_operator = np.zeros( (3, num_neutrinos, num_neutrinos) )

    else :
        cases["a"] = a_magnitude_eV * np.diag(flavor_structure)
        null_operator = np.zeros( (num_neutrinos, num_neutrinos) )


    #
    # Loop over cases
    #


    # Loop over cases
    for case_label, (a_eV) in cases.items() :

        # Report
        print("")
        print("Model : %s" % case_label)


        #
        # Plot oscillations vs RA
        #

        # Define neutrino
        ra_values_deg = np.linspace(0., 360., num=51)
        dec_values_deg = np.linspace(-90., 90., num=19)
        time = "July 15, 2020, 14:30"

        print(dec_values_deg)

        P_shape = (3,len(dec_values_deg), len(ra_values_deg))
        P_std_deimos, P_sme_deimos = np.zeros(P_shape), np.zeros(P_shape)
        P_std_nusquids, P_sme_nusquids = np.zeros(P_shape), np.zeros(P_shape)

        # Loop over dec
        for i in range(len(dec_values_deg)) :

            # Units
            dec_deg = dec_values_deg[i]
            dec_rad = np.deg2rad(dec_values_deg[i])

            # Loop over RA
            for j in range(len(ra_values_deg)) :

                # Units
                ra_deg = ra_values_deg[j]
                ra_rad = np.deg2rad(ra_values_deg[j])

                # Get coszen/azimuth from RA/dec
                coszen, _, azimuth_deg = deimos_calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg, dec_deg=dec_deg, time=time)

                distance_km = calc_path_length_from_coszen(coszen)

                # Define args to osc prob calc
                calc_kw = {
                    "initial_flavor":initial_flavor,
                    "nubar" : nubar,
                    "energy_GeV":E_GeV,
                    "distance_km":distance_km,
                    "ra_rad":ra_rad,
                    "dec_rad":dec_rad,
                }

                # Get std osc probs
                deimos_calculator.set_std_osc()
                result_std = deimos_calculator.calc_osc_prob(**calc_kw)
                P_std_deimos[0,i,j] = np.squeeze(result_std)[0]       #nue
                P_std_deimos[1,i,j] = np.squeeze(result_std)[1]       #numu 
                P_std_deimos[2,i,j] = np.squeeze(result_std)[2]       #nutau 

                nusquids_calculator.set_std_osc()
                result_std = nusquids_calculator.calc_osc_prob(**calc_kw)
                P_std_nusquids[0,i,j] = np.squeeze(result_std)[0]       #nue
                P_std_nusquids[1,i,j] = np.squeeze(result_std)[1]       #numu
                P_std_nusquids[2,i,j] = np.squeeze(result_std)[2]       #nutau
            


                # Get LIV osc probs
                deimos_calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=null_operator, e=null_operator)
                result_sme = deimos_calculator.calc_osc_prob(**calc_kw)
                P_sme_deimos[0,i,j] = np.squeeze(result_sme)[0]       #nue
                P_sme_deimos[1,i,j] = np.squeeze(result_sme)[1]       #numu
                P_sme_deimos[2,i,j] = np.squeeze(result_sme)[2]       #nutau

                nusquids_calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=null_operator, e=null_operator)
                result_sme = nusquids_calculator.calc_osc_prob(**calc_kw)
                P_sme_nusquids[0,i,j] = np.squeeze(result_sme)[0]       #nue
                P_sme_nusquids[1,i,j] = np.squeeze(result_sme)[1]       #numu
                P_sme_nusquids[2,i,j] = np.squeeze(result_sme)[2]       #nutau

        

    # Numpy-ify
    # P_std = np.array(P_std)
    # P_liv = np.array(P_liv)
    # assert P_shape == P_std.shape
    # assert P_shape == P_liv.shape


    dec_index = 12

    # Make fig
    fig, ax = plt.subplots(1,2, figsize=(10, 4) )

    fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ %0.3g deg // %s" % (E_GeV, dec_values_deg[dec_index], time), fontsize=12 )
    
    # Plot osc probs
    ax[0].plot(ra_values_deg, P_std_deimos[0,dec_index,:], color="C1", alpha=0.5,linestyle="-", label="Deimos nue Std. osc.")
    ax[0].plot(ra_values_deg, P_sme_deimos[0,dec_index,:], color="C1", linestyle="--", label="Deimos nue SME osc.")

    ax[0].plot(ra_values_deg, P_std_deimos[1,dec_index,:], color="C2", alpha=0.5, linestyle="-", label="Deimos numu Std. osc.")
    ax[0].plot(ra_values_deg, P_sme_deimos[1,dec_index,:], color="C2", linestyle="--", label="Deimos numu SME osc.")

    ax[0].plot(ra_values_deg, P_std_deimos[2,dec_index,:], color="C3", alpha=0.5, linestyle="-", label="Deimos nutau Std. osc.")
    ax[0].plot(ra_values_deg, P_sme_deimos[2,dec_index,:], color="C3", linestyle="--", label="Deimos nutau SME osc.")
    
    ax[1].plot(ra_values_deg, P_std_nusquids[0,dec_index,:], color="C1", alpha=0.5,linestyle="-", label="Nusquids nue Std. osc.")
    ax[1].plot(ra_values_deg, P_sme_nusquids[0,dec_index,:], color="C1", linestyle="--", label="Nusquids nue SME osc.")

    ax[1].plot(ra_values_deg, P_std_nusquids[1,dec_index,:], color="C2", alpha=0.5, linestyle="-", label="Nusquids numu Std. osc.")
    ax[1].plot(ra_values_deg, P_sme_nusquids[1,dec_index,:], color="C2", linestyle="--", label="Nusquids numu SME osc.")

    ax[1].plot(ra_values_deg, P_std_nusquids[2,dec_index,:], color="C3", alpha=0.5, linestyle="-", label="Nusquids nutau Std. osc.")
    ax[1].plot(ra_values_deg, P_sme_nusquids[2,dec_index,:], color="C3", linestyle="--", label="Nusquids nutau SME osc.")



    #TODO add second x axis with coszen 

    # Formatting
    for i in range(len(ax)):
        ax[i].set_xlabel("RA [deg]")
        ax[i].set_ylabel( "$%s$" % deimos_calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )
        ax[i].set_ylim(-0.01, 1.01)
        ax[i].set_xlim(ra_values_deg[0], ra_values_deg[-1])
        ax[i].grid(True)
        ax[i].legend(fontsize=12)
    fig.tight_layout()


    
    # plot RA vs dec oscillogram

    fig, ax = plt.subplots(3,2, figsize=(9, 10))
    ax = ax.flatten()

    cmap = "seismic"
    vmin, vmax = -1e-6, 1e-6

    fig.suptitle( r"$E$ = %0.3g GeV // %s" % (E_GeV, time), fontsize=12 )
    ax[1].imshow(P_sme_deimos[0,:,:]-P_sme_nusquids[0,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].imshow(P_std_deimos[0,:,:]-P_std_nusquids[0,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax[1].set(xlabel="RA[deg]",ylabel="DEC[deg]",title="SME osc.")
    ax[0].set(xlabel="RA[deg]",ylabel="DEC[deg]",title="Standard osc.")
    cbar0 = fig.colorbar(ax[1].images[0], ax=ax[1], orientation="vertical", fraction=0.05, pad=0.05)
    cbar0.set_label("$%s$" % deimos_calculator.get_transition_prob_tex(initial_flavor, 0, nubar)+":  deimos-nusquids", fontsize=12)

    ax[3].imshow(P_sme_deimos[1,:,:]-P_sme_nusquids[1,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].imshow(P_std_deimos[1,:,:]-P_std_nusquids[1,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax[3].set(xlabel="RA[deg]",ylabel="DEC[deg]")
    ax[2].set(xlabel="RA[deg]",ylabel="DEC[deg]")
    cbar1 = fig.colorbar(ax[3].images[0], ax=ax[3], orientation="vertical", fraction=0.05, pad=0.05)
    cbar1.set_label("$%s$" % deimos_calculator.get_transition_prob_tex(initial_flavor, 1, nubar)+":  deimos-nusquids", fontsize=12)

    ax[5].imshow(P_sme_deimos[2,:,:]-P_sme_nusquids[2,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax[4].imshow(P_std_deimos[2,:,:]-P_std_nusquids[2,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax[5].set(xlabel="RA[deg]",ylabel="DEC[deg]")
    ax[4].set(xlabel="RA[deg]",ylabel="DEC[deg]")
    cbar2 = fig.colorbar(ax[5].images[0], ax=ax[5], orientation="vertical", fraction=0.05, pad=0.05)
    cbar2.set_label("$%s$" % deimos_calculator.get_transition_prob_tex(initial_flavor, 2, nubar)+":  deimos-nusquids", fontsize=12)


    fig.tight_layout()


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
