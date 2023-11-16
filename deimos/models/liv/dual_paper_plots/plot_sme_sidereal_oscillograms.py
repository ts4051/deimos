'''
Plot neutrino oscillograms with sidereal SME parameters activated

Simon Hilding-Nørkjær
'''



import sys, os, collections, datetime
from astropy.time import Time
import numpy as np

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import *


#
# Main
#

if __name__ == "__main__" :


    #
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = 50
    coszen = -1.

   


    #
    # Create model
    #

    # Choose solver
    solver = "deimos" # deimos nusquids
    # solver = "nusquids"

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = np.geomspace(E_GeV[0], E_GeV[-1], num=100)

    # Create calculator
    calculator = OscCalculator(
        tool=solver,
        atmospheric=False,
        num_neutrinos=3,
        **kw
    )

    # Use vacuum
    calculator.set_matter("vacuum")

    # Define detector position
    detector_name = "icecube" # IceCube DUNE
    calculator.set_detector(detector_name)


    #
    # Define physics cases
    #


    directional = True
    sme_basis = "mass"

    # Define LIV coefficients as matrices               [direction_layer],[flavor_row],[flavor_col]
    a_eV = np.zeros( (3,3,3) )
    c = np.zeros( (3,3,3) )
    e = np.zeros( (3,3,3) )

    # set a_eV_y = 1e-22    
    a_eV[2,2,2] = 2e-13



    # Define neutrino
    ra_values_deg = np.linspace(0., 360., num=64)
    dec_values_deg = np.linspace(-90., 90., num=64)
    time = "July 15, 2020, 14:30"

    resolution = 2
    npix = hp.nside2npix(nside=resolution)
    ra_values_deg, dec_values_deg = hp.pix2ang(nside=resolution, ipix=np.arange(npix), lonlat=True)
    


    # print(dec_values_deg)

    P_shape = (3,len(dec_values_deg), len(ra_values_deg))
    P_std, P_liv = np.zeros(P_shape), np.zeros(P_shape)

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
            coszen, _, azimuth_deg = calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg, dec_deg=dec_deg, time=time)

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
            calculator.set_std_osc()
            result_std = calculator.calc_osc_prob(**calc_kw)
            P_std[0,i,j] = np.squeeze(result_std)[0]       #nue
            P_std[1,i,j] = np.squeeze(result_std)[1]       #numu 
            P_std[2,i,j] = np.squeeze(result_std)[2]       #nutau 


            # Get LIV osc probs
            calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=c, e=e)
            result_sme = calculator.calc_osc_prob(**calc_kw)
            P_liv[0,i,j] = np.squeeze(result_sme)[0]       #nue
            P_liv[1,i,j] = np.squeeze(result_sme)[1]       #numu
            P_liv[2,i,j] = np.squeeze(result_sme)[2]       #nutau

        

    # Numpy-ify
    P_std = np.array(P_std)
    P_liv = np.array(P_liv)
    assert P_shape == P_std.shape
    assert P_shape == P_liv.shape
















    # import numpy as np
    # import healpy as hp
    # import matplotlib.pyplot as plt

    # def generate_mock_data(nside=65):
    #     """
    #     Generate mock data for demonstration purposes.
    #     Replace this function with your actual data generation logic.
    #     """
    #     npix = hp.nside2npix(nside)
    #     mock_data = np.random.rand(npix)
    #     return mock_data

    # def plot_oscillogram_on_mollview(data, title="Oscillogram on Mollview"):
    #     """
    #     Plot an oscillogram of P(RA, DEC) on a Mollweide equatorial skymap.
    #     """
    #     nside = hp.npix2nside(len(data))
        
    #     # Check if the number of pixels is consistent with the expected value
    #     if len(data) != 12 * nside**2:
    #         raise ValueError("Invalid number of pixels for the given nside")

    #     theta, phi = hp.pix2ang(nside, range(len(data)))

    #     # Create a Mollweide plot
    #     hp.mollview(data, title=title, cmap='viridis', rot=(0, 90, 0), min=0, max=1)

    #     # Add grid and colorbar
    #     hp.graticule()
    #     # plt.colorbar()

    #     # Show the plot
    #     plt.show()


    # # Generate mock data (replace this with your actual data)
    # data = generate_mock_data()

    # # Plot oscillogram on Mollview
    # plot_oscillogram_on_mollview(P_liv[2,:,:].flatten(), title="Oscillogram on Mollview")


    import numpy as np
    import healpy as hp
    import matplotlib.pyplot as plt

    # ... (Your existing code above) ...

    # Plot oscillogram on Mollview
    def plot_oscillogram_on_mollview(data, title="Oscillogram on Mollview"):
        """
        Plot an oscillogram of P(RA, DEC) on a Mollweide equatorial skymap.
        """
        nside = hp.npix2nside(len(data))
        
        # Check if the number of pixels is consistent with the expected value
        if len(data) != 12 * nside**2:
            raise ValueError("Invalid number of pixels for the given nside")

        theta, phi = hp.pix2ang(nside, range(len(data)))

        # Create a Mollweide plot
        hp.mollview(data, title=title, cmap='viridis', rot=(0, 90, 0), min=0, max=1)

        # Add grid and colorbar
        hp.graticule()
        plt.colorbar()

        # Show the plot
        plt.show()

    # Flatten the P_liv array and plot the oscillogram on Mollview
    plot_oscillogram_on_mollview(P_liv[2,:,:].flatten(), title="Oscillogram on Mollview")

    # ... (Rest of your code) ...




   

    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )