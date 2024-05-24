'''
Compare IceCube with an off-axis detector in neutrino RA,DEC oscillograms for probabilities with sidereal SME parameters and matter effects activated

Simon Hilding-Nørkjær
'''



import sys, os, collections, datetime
from astropy.time import Time
import time as time_module
import numpy as np

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import * #calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import * #get_right_ascension_and_declination
from deimos.utils.constants import * 


#
# Main
#

if __name__ == "__main__" :


    #
    # Define basic system
    #
    off_axis_detector = "Arca"     # "arca" or "dune" or "equator"

    initial_flavor = 1          # numu survival
    nubar = False             # neutrino or antineutrino

    E_array_type =True
    E_GeV = np.array([10000.,20000.])
    # E_GeV = 10000.
    E_node = 0


    atmospheric = True
    sme_basis = "mass"

    a_magnitude_eV = 2e-13 # Overall strength of a component
    c_magnitude = 1e-26 # Overall strength of c component

    ra_dec_grid = [50,50]
    print("Number of points: %d" % (ra_dec_grid[0]*ra_dec_grid[1]))



    # Choose solver (nusquids or deimos)
    solver = "nusquids"

    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    IC_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    Off_axis_calculator = OscCalculator(tool=solver ,atmospheric=atmospheric,**kw)


    # IC_calculator.set_matter(matter_model, matter_kwargs)
    IC_calculator.set_matter("earth")
    IC_calculator.set_detector("icecube")

    Off_axis_calculator.set_matter("earth")
    Off_axis_calculator.set_detector(off_axis_detector)


   

############################ SME PARAMETERS #########################################

    flavor_structure = np.array([0.0, 0.0, 1.0]) # numu->nutau
    direction_structure = np.array([0.0,1.0,0.0]) # Orientation of field
    direction_dec = np.array([0,90,0]) @ direction_structure
    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in direction_structure ])
    null_operator = np.zeros( (3, 3, 3) )





############################# EARTH LAYER BOUNDARIES #########################################

    # Define RA/dec grid (0.0587s per point)  208*202*0.0587s = 41min 7s
    # print("Calculation time estimate: %0.2f minutes" % (ra_dec_grid[0]*ra_dec_grid[1]*0.0587/60.))
    ra_values_deg = np.linspace(0., 360., num=ra_dec_grid[0])
    dec_values_deg = np.linspace(-90., 90., num=ra_dec_grid[1])
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    time = "July 16, 1999, 10:00"

    # horizontal lines
    azimuth = np.linspace(0,360,1000)
    IC_RA_horizon, IC_DEC_horizon = IC_calculator.detector_coords.get_right_ascension_and_declination(0,azimuth,time)
    Off_axis_RA_horizon, Off_axis_DEC_horizon = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(0,azimuth,time)

    # Order detector horizon arrays as RA-ascending for off-axis detector
    Off_axis_indices=np.argsort(Off_axis_RA_horizon)
    Off_axis_RA_horizon = Off_axis_RA_horizon[Off_axis_indices]
    Off_axis_DEC_horizon = Off_axis_DEC_horizon[Off_axis_indices]

    # layer radii and thicknesses from the simple earth model: Originally from PREM paper.
    inner_core_radius = 1221.5
    outer_core_radius = 3480.0
    mantle_radius = 5701.0
    earth_radius_km = 6371.0

    inner_core_thickness_km = inner_core_radius*2                               #x2 because 1221 is the radius
    outer_core_thicknes_km = outer_core_radius - inner_core_radius              #3480 is the outer core radius
    mantle_thickness_km = mantle_radius - outer_core_radius                     #5701 is the mantle radius
    transition_and_crust_thickness_km = earth_radius_km - mantle_radius         #6371 is the earth radius

    # currently transition and crust layer is treated as part of the mantle
    mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km


    # Earth core boundary lines (inner core radius 1216km. https://www.ucl.ac.uk/seismin/explore/Earth.html) 
    pathlength_inner_core = 2*np.sqrt(earth_radius_km**2 - inner_core_radius**2)                 #translate core radius to pathlenght
    cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
    IC_RA_core, IC_DEC_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core,azimuth,time) #Wrong sign in coord transformation
    Off_axis_RA_core, Off_axis_DEC_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core,azimuth,time)

    # Earth Outer core boundary lines (outer core radius 3486km)
    pathlength_outer_core = 2*np.sqrt(earth_radius_km**2 - outer_core_radius**2)
    cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
    IC_RA_outer_core, IC_DEC_outer_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core,azimuth,time)
    Off_axis_RA_outer_core, Off_axis_DEC_outer_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core,azimuth,time)

    # Earth mantle boundary lines (mantle radius 5701km)
    pathlength_mantle = 2*np.sqrt(earth_radius_km**2 - mantle_radius**2)
    cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
    IC_RA_mantle, IC_DEC_mantle = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle,azimuth,time)
    Off_axis_RA_mantle, Off_axis_DEC_mantle = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle,azimuth,time)

    # Order detector mantle arrays as RA-ascending for off-axis detector
    Off_axis_indices=np.argsort(Off_axis_RA_mantle)
    Off_axis_RA_mantle = Off_axis_RA_mantle[Off_axis_indices]
    Off_axis_DEC_mantle = Off_axis_DEC_mantle[Off_axis_indices]






    #
    # MAIN LOOP
    #

    # print(dec_values_deg)
    P_shape = (3,len(dec_values_deg), len(ra_values_deg))
    P_IC, P_Off_axis = np.zeros(P_shape), np.zeros(P_shape)

    ra_dec_shape = (len(dec_values_deg), len(ra_values_deg))
    cosz_IC, cosz_Off_axis = np.zeros(ra_dec_shape), np.zeros(ra_dec_shape)

    t_init = time_module.time()

    # Loop over dec
    for i , dec_rad in enumerate(dec_values_rad) : 

        # Loop over RA
        for j, ra_rad in enumerate(ra_values_rad) :

            # Calculation timing, estimation and progress
            if i==0 and j==0:
                start_time = time_module.time()
                t = 0
            t += 1

            print("Progress: %0.2f%%" % (100.*(i*len(ra_values_rad)+j)/(len(dec_values_rad)*len(ra_values_rad))), end="\r")



            #
            # Calculate oscillation probabilities:
            #


            # Get coszen/azimuth from RA/dec
            IC_cosz_value, _, IC_azimuth_deg = IC_calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_values_deg[j], dec_deg=dec_values_deg[i], time=time)
            Off_axis_cosz_value, _, Off_axis_azimuth_deg = Off_axis_calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_values_deg[j], dec_deg=dec_values_deg[i], time=time)
            

            # Define args to osc prob calc
            calc_kw = {
                "initial_flavor":initial_flavor,
                "nubar" : nubar,
                "energy_GeV":E_GeV,
                "ra_rad":ra_rad,
                "dec_rad":dec_rad,
                "time":time,
            }



            # Get LIV osc probs
            # IC_calculator.set_sme(basis=sme_basis, a_eV=a_eV, c=ct, e=null_operator)
            # Off_axis_calculator.set_sme(basis=sme_basis, a_eV=a_eV, c=ct, e=null_operator)


            P_IC_results, coszen_values_IC, azimuth_values_IC = IC_calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV,c=ct, **calc_kw)
            P_Off_axis_results, coszen_values_Off_axis, azimuth_values_Off_axis = Off_axis_calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, c=ct, **calc_kw)


            # if atmospheric : # Atmospheric takes only coszen
            #     IC_result_sme = IC_calculator.calc_osc_prob(coszen=IC_cosz_value,**calc_kw)
            #     Off_axis_result_sme = Off_axis_calculator.calc_osc_prob(coszen=Off_axis_cosz_value,**calc_kw)
            # else : #non-atmospheric takes only baseline
            #     IC_distance_km = calc_path_length_from_coszen(IC_cosz_value)
            #     Off_axis_distance_km = calc_path_length_from_coszen(Off_axis_cosz_value)
            #     IC_result_sme = IC_calculator.calc_osc_prob(distance_km=IC_distance_km,**calc_kw)
            #     Off_axis_result_sme = Off_axis_calculator.calc_osc_prob(distance_km=Off_axis_distance_km,**calc_kw)


            if E_array_type:
                P_IC[0,i,j] = np.squeeze(P_IC_results)[E_node][0]       #nue
                P_IC[1,i,j] = np.squeeze(P_IC_results)[E_node][1]       #numu
                P_IC[2,i,j] = np.squeeze(P_IC_results)[E_node][2]       #nutau

                P_Off_axis[0,i,j] = np.squeeze(P_Off_axis_results)[E_node][0]
                P_Off_axis[1,i,j] = np.squeeze(P_Off_axis_results)[E_node][1]
                P_Off_axis[2,i,j] = np.squeeze(P_Off_axis_results)[E_node][2]
            # else: 
            #     P_IC[0,i,j] = np.squeeze(IC_result_sme)[0]
            #     P_IC[1,i,j] = np.squeeze(IC_result_sme)[1]
            #     P_IC[2,i,j] = np.squeeze(IC_result_sme)[2]

            #     P_Off_axis[0,i,j] = np.squeeze(Off_axis_result_sme)[0]
            #     P_Off_axis[1,i,j] = np.squeeze(Off_axis_result_sme)[1]
            #     P_Off_axis[2,i,j] = np.squeeze(Off_axis_result_sme)[2]


            # Check that probabilities sum to 1
            assert np.isclose( np.sum(P_IC[:,i,j]), 1.0, atol=1e-10)
            assert np.isclose( np.sum(P_Off_axis[:,i,j]), 1.0, atol=1e-10)
            
            # Save coszen values
            cosz_IC[i,j] = IC_cosz_value
            cosz_Off_axis[i,j] = Off_axis_cosz_value

            # Timing
            if t== 100:
                delta_time = (time_module.time() - start_time)/100.
                print("Calculation time for one iteration: %0.4f seconds" % delta_time)
                print("Total calculation time estimate: %0.2f minutes" % (delta_time*len(dec_values_deg)*len(ra_values_deg)/60.))

    print("Total calculation time: %0.2f minutes" % ((time_module.time()-t_init)/60.) )


    


    # Numpy-ify
    P_IC = np.array(P_IC)
    P_Off_axis = np.array(P_Off_axis)
    assert P_shape == P_IC.shape
    assert P_shape == P_Off_axis.shape



    
    # # plot RA vs dec oscillogram
    linewidth = 2
    alpha = 1

    fig, ax = plt.subplots(3,2, figsize=(9, 10), sharex=True, sharey=True)
    fig.subplots_adjust(right=0.88, wspace=0.045, hspace=0.05)

    ax = ax.flatten()

    # fig.suptitle( r"$E$ = %0.3g GeV // time: %s // SME: a_eV_y=%0.3g // c=%0.3g" % (E_GeV[E_node], time,a_magnitude_eV,c_magnitude), fontsize=12 )
    # ax[0].set_title("IceCube",fontsize=12)
    # ax[1].set_title(off_axis_detector,fontsize=12)

    

    ax[0].imshow(P_IC[0,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)
    ax[1].imshow(P_Off_axis[0,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)

    cbar_ax1 = fig.add_axes([0.90, 0.632, 0.02, 0.247])
    cbar1 = fig.colorbar(ax[1].images[0], cax=cbar_ax1, orientation="vertical", fraction=0.05, pad=0.05)
    cbar1.set_label("$%s$" % Off_axis_calculator.get_transition_prob_tex(initial_flavor, 0, nubar), fontsize=14)
    cbar1.ax.tick_params(labelsize=14)
    cbar1.ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    cbar1.ax.set_yticklabels(["0\n","0.2","0.4","0.6","0.8","\n1.0"])


    ax[2].imshow(P_IC[1,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)
    ax[3].imshow(P_Off_axis[1,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)

    cbar_ax1 = fig.add_axes([0.90, 0.37, 0.02, 0.247])
    cbar1 = fig.colorbar(ax[1].images[0], cax=cbar_ax1, orientation="vertical", fraction=0.05, pad=0.05)
    cbar1.set_label("$%s$" % Off_axis_calculator.get_transition_prob_tex(initial_flavor, 1, nubar), fontsize=14)
    cbar1.ax.tick_params(labelsize=14)
    cbar1.ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    cbar1.ax.set_yticklabels(["0\n","0.2","0.4","0.6","0.8","\n1.0"])

    ax[4].imshow(P_IC[2,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)
    ax[5].imshow(P_Off_axis[2,:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)

    cbar_ax1 = fig.add_axes([0.90, 0.108, 0.02, 0.247])
    cbar1 = fig.colorbar(ax[1].images[0], cax=cbar_ax1, orientation="vertical", fraction=0.05, pad=0.05)
    cbar1.set_label("$%s$" % Off_axis_calculator.get_transition_prob_tex(initial_flavor, 2, nubar), fontsize=14)
    cbar1.ax.tick_params(labelsize=14)
    cbar1.ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    cbar1.ax.set_yticklabels(["0\n","0.2","0.4","0.6","0.8","\n1.0"])





    for i in range(len(ax)-1):

        ax[i].set_xticks([0,90,180,270,360])
        ax[i].set_yticks([-90,-45,0,45,90])
        ax[i].set_yticklabels(["-90\n","-45","0","45","\n90"])
        ax[i].tick_params(labelsize=14)
        ax[5].tick_params(labelsize=14)

        if i%2==0:      #IceCube plots (left column)
            ax[i].plot(IC_RA_horizon,IC_DEC_horizon,color="lime",alpha=alpha,lw=linewidth)#, label="Horizon")
            ax[i].plot(IC_RA_core,IC_DEC_core,color="red",alpha=alpha,lw=linewidth)#, label="Earth inner core")
            ax[i].plot(IC_RA_outer_core,IC_DEC_outer_core,color="orange",alpha=alpha,lw=linewidth)#, label="Earth outer core")
            ax[i].plot(IC_RA_mantle,IC_DEC_mantle,color="yellow",alpha=alpha,lw=linewidth)#, label="Earth mantle")
            #add text to plot
            ax[i].text(0.04, 0.12, "IceCube", transform=ax[i].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax[i].set_ylabel("DEC[deg]",fontsize=14)

        
        else:          #Off-axis plots (right column)
            ax[i].plot(Off_axis_RA_horizon,Off_axis_DEC_horizon,color="lime",alpha=alpha,lw=linewidth)#, label="Horizon")
            ax[i].plot(Off_axis_RA_core,Off_axis_DEC_core,color="red",alpha=alpha,lw=linewidth,linestyle=None)#, label="Earth inner core")
            ax[i].plot(Off_axis_RA_outer_core,Off_axis_DEC_outer_core,color="orange",alpha=alpha,lw=linewidth,linestyle=None)#, label="Earth outer core")
            ax[i].plot(Off_axis_RA_mantle,Off_axis_DEC_mantle,color="yellow",alpha=alpha,lw=linewidth)#, label="Earth mantle")
            ax[i].text(0.04, 0.95, "ARCA", transform=ax[i].transAxes, fontsize=14,color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax[i].plot(90,0, markerfacecolor="gold", markeredgecolor="black",marker="D", markersize=7, linestyle="None")#, label="LIV-field direction")

        # ax[i].plot(-35,280, markerfacecolor="gold", markeredgecolor="black",marker="D", markersize=6, linestyle="None")
        
        # ax[i].legend(fontsize=8)

    ax[5].plot(90,0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=6, linestyle="None", label="LIV-field direction")
    ax[5].plot(Off_axis_RA_horizon,Off_axis_DEC_horizon,color="lime",alpha=alpha,lw=linewidth, label="Horizon")
    ax[5].plot(Off_axis_RA_mantle,Off_axis_DEC_mantle,color="yellow",alpha=alpha,lw=linewidth, label="Mantle")
    ax[5].plot(Off_axis_RA_outer_core,Off_axis_DEC_outer_core,color="orange",alpha=alpha,lw=linewidth,linestyle=None, label="Outer core")
    ax[5].plot(Off_axis_RA_core,Off_axis_DEC_core,color="red",alpha=alpha,lw=linewidth,linestyle=None, label="Inner core")
    ax[5].text(0.04, 0.95, "ARCA", transform=ax[5].transAxes, fontsize=14, color="white",verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    # fig.subplots_adjust({0.0,0.0,0.0,0.0})
    fig.legend(loc="upper center", fontsize=12, ncol=5, bbox_to_anchor=(0.5, 0.93))#,fancybox=True, shadow=True)
# loc=(-1.2,2.10)
    ax[4].set_xlabel("RA[deg]",fontsize=14)
    ax[5].set_xlabel("RA[deg]",fontsize=14)
    ax[4].set_xticklabels([" 0","90","180","270","360     "])   
    ax[5].set_xticklabels([" 0","90","180","270","360     "])

    




    # fig.tight_layout()





    
    # plot RA vs dec oscillogram

    # fig, ax = plt.subplots(1,2, figsize=(9,4))
    # ax = ax.flatten()

    # fig.suptitle( r"Sidereal dependence of $cos(\theta_{zenith})$ in RA,DEC-oscillogram"+f", Time: {time} UTC", fontsize=12 )
    # ax[0].imshow(cosz_IC[:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdYlBu", vmin=-1., vmax=1.)
    # ax[1].imshow(cosz_Off_axis[:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdYlBu", vmin=-1., vmax=1.)

    # #plot horizontal lines 
    # ax[0].plot(IC_RA_horizon,IC_DEC_horizon,color="black",lw=linewidth, label="Horizon")
    # ax[1].plot(Off_axis_RA_horizon,Off_axis_DEC_horizon,color="black",lw=linewidth, label="Horizon")
    # #plot earth core boundary
    # ax[0].plot(IC_RA_core,IC_DEC_core,color="red",lw=linewidth, label="Inner core")
    # ax[1].plot(Off_axis_RA_core,Off_axis_DEC_core,color="red",lw=linewidth, label="Inner core")
    # #plot earth outer core boundary
    # ax[0].plot(IC_RA_outer_core,IC_DEC_outer_core,color="orange",lw=linewidth, label="Outer core")
    # ax[1].plot(Off_axis_RA_outer_core,Off_axis_DEC_outer_core,color="orange",lw=linewidth, label="Outer core")
    # #plot earth mantle boundary
    # ax[0].plot(IC_RA_mantle,IC_DEC_mantle,color="yellow",lw=linewidth, label="Mantle")
    # ax[1].plot(Off_axis_RA_mantle,Off_axis_DEC_mantle,color="yellow",lw=linewidth, label="Mantle")



    # for i in range(len(ax)):
    #     #plot ra,dec=0,0
    #     # ax[i].plot(direction_dec,0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=6, linestyle="None",label="LIV-field direction")
    #     ax[0].legend(fontsize=10)


    # cbar_ms = 4
    
    # ax[0].set(xlabel="RA[deg]",ylabel="DEC[deg]",title="IceCube")
    # ax[1].set(xlabel="RA[deg]",ylabel="DEC[deg]",title=off_axis_detector)
    # cbar0 = fig.colorbar(ax[1].images[0], ax=ax[1], orientation="vertical", fraction=0.05, pad=0.05)
    # cbar0.set_label(r"$Cos(\theta_z)$", fontsize=12)


    #TODO: Figure out why these cosz are not the same as the ones in the plot
    # cbar0.ax.plot(0.5, cosz_inner_core, color="red", marker="o", markersize=cbar_ms, linestyle="None",)
    # cbar0.ax.plot(0.5, cosz_outer_core, color="orange", marker="o", markersize=cbar_ms, linestyle="None",)
    # cbar0.ax.plot(0.5, cosz_mantle, color="yellow", marker="o", markersize=cbar_ms, linestyle="None",)
    # cbar0.ax.plot(0.5, 0, color="black", marker="o", markersize=cbar_ms, linestyle="None",)

    # print("Coszen values:")
    # print(f"Inner core: {cosz_inner_core}")
    # print(f"Outer core: {cosz_outer_core}")
    # print(f"Mantle: {cosz_mantle}")
    # print(f"Horizon: {0}")


    # fig.tight_layout()

    plt.savefig(__file__.replace(".py",".pdf") , bbox_inches="tight")





    #
    # Done
    #

    # print("")
    # dump_figures_to_pdf( __file__.replace(".py",".pdf") )
