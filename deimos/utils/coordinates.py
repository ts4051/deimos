#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoordTransform: A class for coordinate transformations and path length calculations used for sidereal LIV.

Created on Tue Jul 18 13:35:29 2023
Author: janni
"""

import datetime
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle, ICRS, get_sun
from astropy.time import Time

from deimos.utils.oscillations import calc_path_length_from_coszen
from deimos.utils.constants import DEFAULT_ATMO_PROD_HEIGHT_km

class DetectorCoords(object):
    """
    DetectorCoords class provides methods for coordinate transformations used for e.g. sidereal LIV.

    Parameters:
        detector_lat (str or float): Latitude of the detector in degrees or degrees-minutes-seconds format.
        detector_long (str or float): Longitude of the detector in degrees or degrees-minutes-seconds format.
        detector_height_m (float): Height of the detector above sea level in meters.
    """

    def __init__(self, detector_lat, detector_long, detector_height_m):

        # Convert latitude and longitude to degrees
        # latitude in range [-90,90]
        if isinstance(detector_lat, str):
            detector_lat = Angle(detector_lat)
            detector_lat = detector_lat.deg

        # longitude in range (-180,180)
        if isinstance(detector_long, str):
            detector_long = Angle(detector_long)
            detector_long = detector_long.deg

        # Create Earth location
        self.detector_location = EarthLocation(lat=detector_lat*u.deg, lon=detector_long*u.deg, height=detector_height_m*u.m)

        #TODO extract Earth radius from EarthLocation for use in OscWrapper


    @property
    def detector_depth_m(self):
        return -self.detector_location.height/u.m
    

    def get_time(self, time):
        """
        Create an Astropy Time object.

        Parameters:
            time (str): Date-time string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            astropy.time.Time: Time object representing the provided date string.
        """

        #TODO handle timezone

        # If object is already an astropy.Time object, nothing to do
        if isinstance(time, Time) :
            return time

        # If object is a python datetime object, conversion is easy
        elif isinstance(time, datetime.datetime) :
            return Time(time)

        # If object is a date string, handle the conversion here
        elif isinstance(time, str) :

            # Define the format of the input date string with seconds
            date_time_format = "%B %d, %Y, %H:%M:%S"
            # Try parsing the date string with seconds
            try:
                time_obj = Time.strptime(time, date_time_format)
            except ValueError:
                # If parsing with seconds fails, try parsing without seconds
                date_time_format_no_seconds = "%B %d, %Y, %H:%M"
                try:
                    time_obj = Time.strptime(time, date_time_format_no_seconds)
                except ValueError:
                    raise ValueError("Invalid date format. Please provide a date in the form 'July 15, 2023, 14:30' or 'July 15, 2023, 14:30:25'.")

            # Check output format
            assert isinstance(time_obj, Time)
                            
            return time_obj

        else :
            raise NotImplemented("Unsupported type : %s" % type(time))
        

    #
    # Coord system conversions
    #

    def get_coszen_altitude_and_azimuth(self, ra_rad, dec_rad, time):
        """
        Get the cosine of the zenith angle and the altitude and azimuth corresponding to the given 
        declination and right ascension (at the specified date/time). Depends on the detector location.

        Parameters:
            ra (float): Declination in radians.
            dec (float): Right ascension in radians.
            time (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
            
        Returns:
            triple: Triple containing coszen (cosine of the zenith angle) and azimuth in radians.
        """

        # Create observation time object
        time = self.get_time(time)

        # Create neutrino direction
        # The direction of the vector calculated in spherical coordinates using the values of ra and dec will be opposite to the direction of the neutrinos. 
        # This is taken into account in the calculation of the direction vector in the SME in density_matrix_osc_solver.py. 
        nu_dir = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad, frame='icrs')

        # Convert to local coordinate system (zenith, azimuth)
        # Coordinate transformation to the horizontal coordinate system. Azimuth is oriented East of North (i.e., N=0, E=90 degrees) 
        # Altitude is the angle above the horizon
        frame = AltAz(obstime=time, location=self.detector_location)
        nu_dir_alt_az = nu_dir.transform_to(frame)

        # Get zenith angle and convert to cos(zen)
        coszen = np.cos(nu_dir_alt_az.zen.to(u.rad).value)
        
        # Get altitude angle
        altitude_rad = nu_dir_alt_az.alt.to(u.rad).value
        
        # Get azimuthal angle
        azimuth_rad = nu_dir_alt_az.az.to(u.rad).value
        
        return coszen, altitude_rad, azimuth_rad
    
    
    def get_right_ascension_and_declination(self, coszen, azimuth_rad, time):
        """
        Get the right ascension and declination  corresponding to the given 
        cosine of the zenith angle and azimuth (at the specified date/time). Depends on the detector location.

        Parameters:
            coszen (float): Cosine of the zenith angle.
            azimuth_rad (float): Azimuth angle in radians.
            time (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            tuple: Tuple containing right ascension and declination in radians.
        """

        # Create observation time object
        time = self.get_time(time)
        
        # Convert conszen to altitutde
        zen = np.arccos(coszen)
        alt = np.pi/2 - zen
        
        # Create neutrino direction in AltAz frame
        nu_dir_alt_az = AltAz(az=azimuth_rad*u.rad, alt=alt*u.rad, obstime=time, location=self.detector_location)
            
        # Transform back to ICRS frame (celestial: right ascension & declination)
        icrs_frame_observed = ICRS()
        nu_dir_icrs = nu_dir_alt_az.transform_to(icrs_frame_observed)
    
        # Get right ascension and declination
        ra_rad = nu_dir_icrs.ra.rad
        dec_rad = nu_dir_icrs.dec.rad

        return ra_rad, dec_rad


    #
    # Functions for atmospheric neutrinos
    #

    def calc_path_length_from_coszen(self, coszen, production_height_km=DEFAULT_ATMO_PROD_HEIGHT_km) :
        '''
        Calculate the path length (baseline) in [km] for an atmospheric neutrino from a given coszen, for this detector depth
        '''
        detector_depth_km = self.detector_depth_m.value * 1e-3 #TODO use proper unit handling methods
        L_km = calc_path_length_from_coszen(cz=coszen, h=production_height_km, d=detector_depth_km)
        return L_km


    #
    # Functions for neutrino beams
    #

    def get_coszen_azimuth_for_beam(self, beam_coords, time):
        """
        Get the coszen/azimuth of a particle beam betwen a specified beam location to this detector location.
        """

        #TODO why does this depend on time? both points are fixed on the Earth so rotate together?

        time = self.get_time(time)

        # Get the beam location as a coordinate in a terrestial frame (ITRS)
        beam_itrs = beam_coords.detector_location.get_itrs(obstime=time)

        # Create the detector local observation frame (alt-azimuth)
        frame = AltAz(location=self.detector_location, obstime=time)

        # Transform direction to beam into detector coords
        beam_dir_alt_az = beam_itrs.transform_to(frame)
        coszen = np.cos(beam_dir_alt_az.zen.to(u.rad).value)
        azimuth = beam_dir_alt_az.az.to(u.rad).value

        return coszen, azimuth


    def get_right_ascension_and_declination_for_beam(self, beam_coords, time):
        """
        Get the RA/dec of particle beam from a specified beam location to this detector location, at the specified time
        """

        # First, get direction in local coords
        coszen, azimuth_rad = self.get_coszen_azimuth_for_beam(beam_coords=beam_coords, time=time)

        # Now convert to RA/dec, at the given time
        ra, dec = self.get_right_ascension_and_declination(coszen=coszen, azimuth_rad=azimuth_rad, time=time)

        return ra, dec


    def get_beam_detector_distance(self, beam_coords):
        """
        Get the straight line (not following Earth's surface) between beam and detector (e.g. baseline)
        """

        # Check beam origin coordinates (re-using DetectorCoords class for this)
        assert isinstance(beam_coords, DetectorCoords)

        # Get vector between beam and detector
        dx, dy, dz = self.detector_location.to_geocentric()  #TODO How is this coord system determined? Is azimuth correct below?
        bx, by, bz = beam_coords.detector_location.to_geocentric()
        ux, uy, uz = dx-bx, dy-by, dz-bz

        # Get magnitude
        dist = np.sqrt( np.square(ux) + np.square(uy) + np.square(uz) )

        return dist.to(u.m).value # [m]



#
# Particle- (rather than detector-) based coordinate transforms
#

def get_neutrino_direction_vector(ra_rad, dec_rad) :
    '''
    Get neutrino direction vector in equatorial coordinate system, given its RA/declination
    '''

    # Get unit vector
    c = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad)
    ux, uy, uz = -c.cartesian.x, -c.cartesian.y, -c.cartesian.z # Inverse of apparent direction

    # Test it
    assert np.isclose( np.sqrt( ux**2. + uy**2. + uz**2. ), 1. ), "Direction is not unit vector?"

    return np.array([ux, uy, uz])




#
# Test
#

if __name__ == "__main__" :

    #TODO round trip test fails, investigate...

    from deimos.utils.plotting import *
    import collections


    #
    # Round trip test
    #

    # Init
    det_coords = DetectorCoords(
        detector_lat="89°59′24″S", # IceCube
        detector_long="63°27′11″W",
        detector_height_m=-1400,
    )
        
    # Round trip test: Start from RA/dec and convert to zenith/azimuth, then back again to check we recover the original inputs
    time =  "July 15, 2023, 14:30"
    ra_rad, dec_rad = np.deg2rad(13.), np.deg2rad(47.)
    coszen, altitude, azimuth = det_coords.get_coszen_altitude_and_azimuth(ra_rad=ra_rad, dec_rad=dec_rad, time=time)
    ra_rad2, dec_rad2 = det_coords.get_right_ascension_and_declination(coszen=coszen, azimuth_rad=azimuth, time=time)
    assert np.isclose(ra_rad, ra_rad2), "%s != %s" % (ra_rad, ra_rad2)
    assert np.isclose(dec_rad, dec_rad2), "%s != %s" % (dec_rad, dec_rad2)
    coszen2, _, azimuth2 = det_coords.get_coszen_altitude_and_azimuth(ra_rad=ra_rad2, dec_rad=dec_rad2, time=time)
    assert np.isclose(coszen, coszen2), "%s != %s" % (coszen, coszen2)
    assert np.isclose(azimuth, azimuth2), "%s != %s" % (azimuth, azimuth2)

    print("Round trip test PASSED")


    #
    # Plot conversions
    #

    num_points = 100

    # Define some edge case simple detector locations
    detectors = collections.OrderedDict()
    detectors["South Pole"] = DetectorCoords(
        detector_lat=-90.,
        detector_long=0.,
        detector_height_m=0.,
    )
    detectors["Equator"] = DetectorCoords(
        detector_lat=0.,
        detector_long=0.,
        detector_height_m=0.,
    )
    detectors["Midpoint"] = DetectorCoords(
        detector_lat=-45,
        detector_long=0.,
        detector_height_m=0.,
    )


    # Loop over detector cases
    for det_name, det_coords in detectors.items() :

        # RA/dec vs coszen at fixed time, for different azimuth values
        time = "January 1, 2024, 00:00"
        fig, ax = plt.subplots(nrows=2, figsize=(6,8))
        fig.suptitle("%s, %s" % (det_name, time))
        coszen = np.linspace(-1., +1., num=num_points)
        for azimuth_deg, color, linestyle in zip([0., 90.], ["red", "blue"], ["-", "--"]) :
            ra_rad, dec_rad = det_coords.get_right_ascension_and_declination(coszen=coszen, azimuth_rad=np.deg2rad(azimuth_deg), time=time)
            ra_deg, dec_deg = np.rad2deg(ra_rad), np.rad2deg(dec_rad)
            ax[0].plot(coszen, ra_deg, color=color, linestyle=linestyle, lw=3, label=r"Azimuth = %0.1f [deg]"%azimuth_deg)
            ax[1].plot(coszen, dec_deg, color=color, linestyle=linestyle, lw=3, label=r"Azimuth = %0.1f [deg]"%azimuth_deg)
        ax[0].set_xlabel("coszen")
        ax[0].set_ylabel("RA [deg]")
        ax[0].set_xlim(coszen[0], coszen[-1])
        ax[0].set_ylim(0., 360.)
        ax[1].set_xlabel("coszen")
        ax[1].set_ylabel("Dec [deg]")
        ax[1].set_xlim(coszen[0], coszen[-1])
        ax[1].set_ylim(-91, +91.)
        ax[0].legend()
        ax[0].grid(True)
        ax[1].grid(True)
        fig.tight_layout()

        # coszen/azimuth vs RA at fixed time, for different decination values
        time = "January 1, 2024, 00:00"
        fig, ax = plt.subplots(nrows=2, figsize=(6,6))
        fig.suptitle("%s, %s" % (det_name, time))
        ra_deg = np.linspace(0., 360., num=num_points)
        for dec_deg, color, linestyle in zip([0., 90.], ["red", "blue"], ["-", "--"]) :
            coszen, altitude_rad, azimuth_rad = det_coords.get_coszen_altitude_and_azimuth(ra_rad=np.deg2rad(ra_deg), dec_rad=np.deg2rad(np.full(ra_deg.size, dec_deg)), time=time)
            azimuth_deg = np.deg2rad(azimuth_rad)
            ax[0].plot(ra_deg, coszen, color=color, linestyle=linestyle, lw=3, label=r"$\delta$ = %0.1f [deg]"%dec_deg)
            ax[1].plot(ra_deg, azimuth_deg, color=color, linestyle=linestyle, lw=3, label=r"$\delta$ = %0.1f [deg]"%dec_deg)
        ax[0].set_xlabel("RA [deg]")
        ax[0].set_ylabel("coszen")
        ax[0].set_xlim(ra_deg[0], ra_deg[-1])
        ax[0].set_ylim(-1.01, 1.01)
        ax[1].set_xlabel("RA [deg]")
        ax[1].set_ylabel("azimuth [deg]")
        ax[1].set_xlim(ra_deg[0], ra_deg[-1])
        ax[1].set_ylim(-1., 361.)
        ax[0].legend()
        ax[0].grid(True)
        ax[1].grid(True)
        fig.tight_layout()

        # RA/dec vs time at fixed coszen/azimuth
        start_time = datetime.datetime(2024, 1, 1, 0, 0, 0, 0) # Midnight, Jan 1st 2024
        hr_values = np.linspace(0., 24., num=num_points) # One sidereal day
        time_values = [ start_time + datetime.timedelta(hours=hr)  for hr in hr_values ]
        fig, ax = plt.subplots(nrows=2, figsize=(6,8))
        coszen, azimuth_deg = -1, 0. # Beam on opposite side of Earth 
        fig.suptitle("%s, coszen = %.1f, azimuth = %i deg" % (det_name, coszen, azimuth_deg))
        ra_deg, dec_deg = [], []
        for t in time_values :
            r, d = det_coords.get_right_ascension_and_declination(coszen=coszen, azimuth_rad=np.deg2rad(azimuth_deg), time=t)
            ra_deg.append( np.rad2deg(r) )
            dec_deg.append( np.rad2deg(d) )
        ax[0].plot(hr_values, ra_deg, color="orange", linestyle="-", lw=3)
        ax[1].plot(hr_values, dec_deg, color="orange", linestyle="-", lw=3)
        ax[0].set_xlabel("Hours")
        ax[0].set_ylabel("RA [deg]")
        ax[0].set_xlim(hr_values[0], hr_values[-1])
        ax[0].set_ylim(0., 360.)
        ax[1].set_xlabel("Hours")
        ax[1].set_ylabel("Dec [deg]")
        ax[1].set_xlim(hr_values[0], hr_values[-1])
        ax[1].set_ylim(-91, +91.)
        # ax[0].legend()
        ax[0].grid(True)
        ax[1].grid(True)
        fig.tight_layout()


    #
    # Test beam calculations
    #

    # Latitude scan (Pole)  

    time = Time(datetime.datetime(2024, 1, 1, 0, 0, 0, 0)) # Midnight, Jan 1st 2024

    beam_lat_deg = -90.
    beam_coords = DetectorCoords(detector_lat=beam_lat_deg, detector_long=0., detector_height_m=0.)

    det_lat_values_deg = np.linspace(-90., +90., num=num_points)
    coszen_values, zenith_values_deg, azimuth_deg_values = [], [], []
    for det_lat_deg in det_lat_values_deg :

        # Make det
        det_coords = DetectorCoords(detector_lat=det_lat_deg, detector_long=0., detector_height_m=0.)

        # Get AltAz direction to beam
        coszen, azimuth_rad = det_coords.get_coszen_azimuth_for_beam(beam_coords, time=time)
        zenith_deg = np.rad2deg(np.arccos(coszen))
        azimuth_deg = np.rad2deg(azimuth_rad)

        coszen_values.append(coszen)
        zenith_values_deg.append(zenith_deg)
        azimuth_deg_values.append(azimuth_deg)

    # Plot
    fig, ax = plt.subplots(nrows=3, figsize=(6,12))
    fig.suptitle("Beam at North Pole")
    ax[0].axvline(beam_lat_deg, color="purple", linestyle="--")
    ax[1].axvline(beam_lat_deg, color="purple", linestyle="--")
    ax[2].axvline(beam_lat_deg, color="purple", linestyle="--")
    ax[0].plot(det_lat_values_deg, coszen_values, color="orange")
    ax[1].plot(det_lat_values_deg, zenith_values_deg, color="orange")
    ax[2].plot(det_lat_values_deg, azimuth_deg_values, color="orange")
    ax[0].set_xlabel("Detector latitude [deg]")
    ax[1].set_xlabel("Detector latitude [deg]")
    ax[2].set_xlabel("Detector latitude [deg]")
    ax[0].set_ylabel("coszen")
    ax[1].set_ylabel("zenith [deg]")
    ax[2].set_ylabel("azimuth [deg]")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    fig.tight_layout()



    # Longitude scan (equator) ... 

    beam_lon_deg = 0.
    beam_coords = DetectorCoords(detector_lat=0., detector_long=beam_lon_deg, detector_height_m=0.)

    det_lon_values_deg = np.linspace(0., 360., num=num_points)
    coszen_values, zenith_values_deg, azimuth_deg_values = [], [], []
    for det_lon_deg in det_lon_values_deg :

        # Make det
        det_coords = DetectorCoords(detector_lat=0., detector_long=det_lon_deg, detector_height_m=0.)

        # Create the detector frame
        frame = AltAz(obstime=time, location=det_coords.detector_location)

        # Get AltAz direction to beam
        coszen, azimuth = det_coords.get_coszen_azimuth_for_beam(beam_coords, time=time)#
        zenith_deg = np.rad2deg(np.arccos(coszen))
        azimuth_deg = np.rad2deg(azimuth)

        coszen_values.append(coszen)
        zenith_values_deg.append(zenith_deg)
        azimuth_deg_values.append(azimuth_deg)

    # Plot
    fig, ax = plt.subplots(nrows=3, figsize=(6,12))
    plt.suptitle("Beam at Equator")

    ax[0].axvline(beam_lon_deg, color="purple", linestyle="--")
    ax[1].axvline(beam_lon_deg, color="purple", linestyle="--")
    ax[2].axvline(beam_lon_deg, color="purple", linestyle="--")
    ax[0].plot(det_lon_values_deg, coszen_values, color="orange")
    ax[1].plot(det_lon_values_deg, zenith_values_deg, color="orange")
    ax[2].plot(det_lon_values_deg, azimuth_deg_values, color="orange")
    ax[0].set_xlabel("Detector longitude [deg]")
    ax[1].set_xlabel("Detector longitude [deg]")
    ax[2].set_xlabel("Detector longitude [deg]")
    ax[0].set_ylabel("coszen")
    ax[1].set_ylabel("zenith [deg]")
    ax[2].set_ylabel("azimuth [deg]")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    fig.tight_layout()


    #
    # Tests done
    # 

    # Dump figs
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
