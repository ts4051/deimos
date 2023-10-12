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
        self.observer_frame = None
        self.datetime_obj = None


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
        

    def get_coszen_altitude_and_azimuth(self, ra_deg, dec_deg, time):
        """
        Get the cosine of the zenith angle and the azimuth corresponding to the given 
        declination and right ascension (at the specified date/time). Depends on the detector location.

        Parameters:
            ra_deg (float): Declination in degrees.
            dec_deg (float): Right ascension in degrees.
            time (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            tuple: Tuple containing coszen (cosine of the zenith angle) and azimuth in degrees.
        """

        # Create observation time object
        time = self.get_time(time)

        # Create neutrino direction
        # TODO: should this be the opposite direction (e.g., travel direction vs origin direction?)
        nu_dir = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
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
    
    
    def get_right_ascension_and_declination(self, coszen, azimuth, time):
        """
        Get the right ascension and declination corresponding to the given 
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

        # Create neutrino direction in AltAz frame
        nu_dir_alt_az = AltAz(az=azimuth*u.deg, alt=(np.arccos(coszen)-np.pi/2)*u.rad, obstime=time, location=self.detector_location)
        
        # Transform back to ICRS frame (right ascension, declination)
        icrs_frame_observed = ICRS()
        nu_dir_icrs = nu_dir_alt_az.transform_to(icrs_frame_observed)
    
        # Get right ascension and declination in degrees
        right_ascension = nu_dir_icrs.ra.degree
        declination = nu_dir_icrs.dec.degree
    
        return right_ascension, declination


    # def path_length(self, coszen, prodHeight, detectorDepth):
    #     """
    #     Calculate the path length of neutrinos from the production height to the detector.

    #     Parameters:
    #         coszen (array-like): Cosine of the zenith angle(s) of the neutrinos.
    #         prodHeight (float): Height at which the neutrinos are produced above the Earth's surface (in km).
    #         detectorDepth (float): Depth at which the detector lies below the Earth's surface (in km).

    #     Returns:
    #         numpy.ndarray: An array containing the path lengths of neutrinos in the same shape as coszen. Units: km       
    #     """

    #     #TODO merge with calc_path_length_from_coszen
        
    #     # Approximate Earth's radius in km
    #     rEarth = 6371
        
    #     #Create array of the same form as coszen variable to store pathlengths
    #     path_lengths = np.zeros_like(coszen)
        
    #     #Calculate pathlength
    #     expression = (rEarth - detectorDepth)**2 * (-1 + coszen**2) + (rEarth + prodHeight)**2
    #     # Check for non-negative values
    #     mask = expression >= 0  
    #     path_lengths = -rEarth * coszen + np.sqrt(expression[mask])
        
    #     return path_lengths



#
# Test
#

if __name__ == "__main__" :


    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails
    #TODO round trip test fails



    # Init
    det_coords = DetectorCoords(
        detector_lat="89°59′24″S", # IceCube
        detector_long="63°27′11″W",
        detector_height_m=1400,
    )
        
    # Round trip test
    time =  "July 15, 2023, 14:30"
    ra_deg, dec_deg = 13., 47.
    coszen, altitude, azimuth = det_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg, dec_deg=dec_deg, time=time)
    ra_deg2, dec_deg2 = det_coords.get_right_ascension_and_declination(coszen=coszen, azimuth=azimuth, time=time)
    assert ra_deg == ra_deg2, "%s != %s" % (ra_deg, ra_deg2)
    assert dec_deg == dec_deg2, "%s != %s" % (dec_deg, dec_deg2)
    coszen2, _, azimuth2 = det_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg2, dec_deg=dec_deg2, time=time)
    assert coszen == coszen2, "%s != %s" % (coszen_deg, coszen_deg2)
    assert azimuth == azimuth2, "%s != %s" % (azimuth, azimuth2)

