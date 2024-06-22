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

        # Init state variables
        # self.observer_frame = None
        # self.datetime_obj = None


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
        

    def get_coszen_altitude_and_azimuth(self, ra_deg, dec_deg, time, deg=True):
        """
        Get the cosine of the zenith angle and the altitude and azimuth (in degrees) corresponding to the given 
        declination and right ascension (at the specified date/time). Depends on the detector location.

        Parameters:
            ra_deg (float): Declination in degrees.
            dec_deg (float): Right ascension in degrees.
            time (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
            deg (boolean): Set to False if ra and dec are provided in rad.
            
        Returns:
            triple: Triple containing coszen (cosine of the zenith angle) and azimuth in degrees.
        """

        # Create observation time object
        time = self.get_time(time)

        # Create neutrino direction
        # The direction of the vector calculated in spherical coordinates using the values of ra and dec will be opposite to the direction of the neutrinos. 
        # This is taken into account in the calculation of the direction vector in the SME in density_matrix_osc_solver.py. 
        if deg == True:
            nu_dir = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        else:
            nu_dir = SkyCoord(ra=ra_deg*u.rad, dec=dec_deg*u.rad, frame='icrs')
        # SkyCoord object that is per default in the icrs frame
        
        # Convert to local coordinate system (zenith, azimuth)
        # Coordinate transformation to the horizontal coordinate system. Azimuth is oriented East of North (i.e., N=0, E=90 degrees) 
        # Altitude is the angle above the horizon
        frame = AltAz(obstime=time, location=self.detector_location)
        nu_dir_alt_az = nu_dir.transform_to(frame)

        # Get zenith angle and convert to cos(zen)
        coszen = np.cos(nu_dir_alt_az.zen.to(u.rad).value)
        
        # Get altitude angle, in degrees
        altitude = nu_dir_alt_az.alt.to(u.degree).value
        
        # Get azimuthal angle, in degrees
        azimuth = nu_dir_alt_az.az.to(u.degree).value
        
        return coszen, altitude, azimuth
    
    
    def get_right_ascension_and_declination(self, coszen, azimuth, time, deg=True):
        """
        Get the right ascension and declination (in degrees)  corresponding to the given 
        cosine of the zenith angle and azimuth (at the specified date/time). Depends on the detector location.

        Parameters:
            coszen (float): Cosine of the zenith angle.
            azimuth (float): Azimuth angle in degrees.
            time (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            tuple: Tuple containing right ascension and declination in degrees.
        """
        # Create observation time object
        time = self.get_time(time)
        
        
        # Convert conszen to altitutde
        zen = np.arccos(coszen)
        alt = np.pi/2 - zen
        
        # Create neutrino direction in AltAz frame
        if deg == True:
            nu_dir_alt_az = AltAz(az=azimuth*u.deg, alt=alt*u.rad, obstime=time, location=self.detector_location)
        else:
            nu_dir_alt_az = AltAz(az=azimuth*u.rad, alt=alt*u.rad, obstime=time, location=self.detector_location)

            
        # Transform back to ICRS frame (right ascension, declination)
        icrs_frame_observed = ICRS()
        nu_dir_icrs = nu_dir_alt_az.transform_to(icrs_frame_observed)
    
        # Get right ascension and declination
        right_ascension = nu_dir_icrs.ra
        declination = nu_dir_icrs.dec
        if deg :
            right_ascension = right_ascension.degree
            declination = declination.degree
        else :
            right_ascension = right_ascension.rad
            declination = declination.rad

        return right_ascension, declination



    def calc_path_length_from_coszen(self, coszen, production_height_km=DEFAULT_ATMO_PROD_HEIGHT_km) :
        '''
        Calculate the path length (baseline) in [km] for an atmospheric neutrino from a given coszen, for this detector depth
        '''
        detector_depth_km = self.detector_depth_m.value * 1e3 #TODO use proper unit handling methods
        L_km = calc_path_length_from_coszen(cz=coszen, h=production_height_km, d=detector_depth_km)
        return L_km


    def get_right_ascension_and_declination_for_beam(self, beam_coords, time):
        """
        Get the RA/dec of a particle beam betwen a specified beam location to this detector location, at the specified time
        """

        # Check beam origin coordinates (re-using DetectorCoords class for this)
        assert isinstance(beam_coords, DetectorCoords)

        # Get vector between beam and detector
        dx, dy, dz = self.detector_location.to_geocentric()  #TODO How is this coord system determined? Is azimuth correct below?
        bx, by, bz = beam_coords.detector_location.to_geocentric()
        ux, uy, uz = dx-bx, dy-by, dz-bz

        # Convert to zenith/azimuth
        azimuth_rad = 0. if ux.value == 0 else np.arctan(uy / ux).to(u.rad).value
        zenith_rad = np.arctan( np.sqrt( np.square(ux) + np.square(uy) ) / uz ).to(u.rad).value
        coszen = np.cos(zenith_rad)

        # Now covert to RA/dec
        ra, dec = self.get_right_ascension_and_declination(coszen=coszen, azimuth=azimuth_rad, time=time, deg=False)

        # print("\n\n")
        # print("Beam : %s" % beam_coords.detector_location)
        # print("Detector : %s" % self.detector_location)
        # print("Beam direction from detector : x, y, z = %s, %s, %s" % (ux, uy, uz))
        # print("Beam direction from detector : coszen, azimuth = %s, %s" % (coszen,  azimuth_rad))
        # print("Beam RA,dec = %s, %s" % (ra, dec))
        # print("\n")

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

        return dist/u.m


def get_neutrino_direction_vector(ra_deg, dec_deg, deg=True) :
    '''
    Get neutrino direction vector in equatorial coordinate system, given its RA/declination
    '''

    # Get unit vector
    if deg == True:
        c = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    else:
        c = SkyCoord(ra=ra_deg*u.rad, dec=dec_deg*u.rad)
    ux, uy, uz = -c.cartesian.x, -c.cartesian.y, -c.cartesian.z    

    # Test it
    assert np.isclose( np.sqrt( ux**2. + uy**2. + uz**2. ), 1. ), "Direction is not unit vector?"

    return np.array([ux, uy, uz])




#
# Test
#

if __name__ == "__main__" :

    #TODO round trip test fails, investigate...

    # Init
    det_coords = DetectorCoords(
        detector_lat="89°59′24″S", # IceCube
        detector_long="63°27′11″W",
        detector_height_m=-1400,
    )
        
    # Round trip test: Start from RA/dec and convert to zenith/azimuth, then back again to check we recover the original inputs
    time =  "July 15, 2023, 14:30"
    ra_deg, dec_deg = 13., 47.
    coszen, altitude, azimuth = det_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg, dec_deg=dec_deg, time=time, deg=True)
    ra_deg2, dec_deg2 = det_coords.get_right_ascension_and_declination(coszen=coszen, azimuth=azimuth, time=time, deg=True)
    assert ra_deg == ra_deg2, "%s != %s" % (ra_deg, ra_deg2)
    assert dec_deg == dec_deg2, "%s != %s" % (dec_deg, dec_deg2)
    coszen2, _, azimuth2 = det_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg2, dec_deg=dec_deg2, time=time, deg=True)
    assert coszen == coszen2, "%s != %s" % (coszen, coszen2)
    assert azimuth == azimuth2, "%s != %s" % (azimuth, azimuth2)

