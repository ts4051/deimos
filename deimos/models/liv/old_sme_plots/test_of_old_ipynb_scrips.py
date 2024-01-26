import sys, os, collections
import numpy as np
from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.coordinates import *

'''
Set detector location and location of the neutrino source on the sky
'''

#Parameters

#Detector location (IceCube)
detector_lat = "89°59′24″S"
detector_long = "63°27′11″W"
detector_height = 0


#Resolution
num =3

#Neutrino source
ra = np.full(num,180)
dec = np.linspace(-90,90,num)
date_str = "July 17, 2022, 23:20"
utc_offset_hr = 0


#Calculate coszen
coordinates = DetectorCoords(
    detector_lat = detector_lat, 
    detector_long = detector_long, 
    detector_height_m = detector_height
)

coszen,_, azimuth = coordinates.get_coszen_altitude_and_azimuth(
    ra_deg = ra,
    dec_deg = dec,
    time = date_str,
)

#print('coszen = ', coszen)


'''
Plot neutrino oscillations @IceCube
'''


#
# Main
#

if __name__ == "__main__" :
    
    
    #
    # Create model
    #

    # Create calculator
    calculator = OscCalculator(
        tool="deimos",
        atmospheric=True,
        num_neutrinos=3,
    )

    # Use vacuum
    calculator.set_matter("vacuum")
    

    #
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1, 10, num=num)
    coszen = coszen
        
    # Calc osc probs and plot, with SME
    a_eV = np.zeros((3,3,3))
    c = np.zeros((3,3,3))
    e = np.zeros((3,3,3))
    # Set flavour independent values
    e[1,:,:] = 1e-9
    calculator.set_sme(
        directional=True,
        basis="mass",
        a_eV=a_eV, c=c, e=e
        )


    #
    # Plot oscillation vs energy
    #
 
    # Calc osc probs and plot, with SME
    
    
    # Set detector location
    # calculator.set_detector_location(
    #     lat = detector_lat, 
    #     long = detector_long, 
    #     height_m = detector_height
    # )

    calculator.set_detector("IceCube")
                                     
    # Set neutrino source
    # calculator.set_neutrino_source(ra_deg = ra,
    #                               dec_deg = dec,
    #                               date_str = date_str)

    calculator.plot_osc_prob_skymap_2D(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, date_str = date_str, resolution=32)
    
    # Save the second figure as a PDF
    # fig.savefig("SME_oscillation.pdf")
    # label=f"SME (a (in eV) = {tuple(f"{val:.3g}" for val in a)}, c = {tuple(f"{val:.3g}" for val in c)})"                                                                                                                                            


    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )

    #
    # Done
    #