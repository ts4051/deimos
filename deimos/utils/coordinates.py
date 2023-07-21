#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some coordinate transformations and the definition of the pathlength used for sidereal LIV
Created on Tue Jul 18 13:35:29 2023

@author: janni
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from datetime import datetime, timedelta
# plt.style.use('/Users/barbaraskrzypek/Documents/IceCube/LVNeutrinos/LV/paper.mplstyle')

class CoordTransform(object):
    
    def __init__(self, detector_lat, detector_long, detector_height_m):
        '''
        Initialize the CoordTransform object with the location of the detector.
        
        Parameters:
            detector_lat (str or float): Latitude of the detector in degrees or degrees-minutes-seconds format.
            detector_long (str or float): Longitude of the detector in degrees or degrees-minutes-seconds format.
            detector_height_m (float): Height of the detector above sea level in meters.
        '''

        # Convert latitude and longitude to degrees
        if isinstance(detector_lat, str):
            detector_lat = Angle(detector_lat)
            detector_lat = detector_lat.deg
        #latitude in range [-90,90]

        if isinstance(detector_long, str):
            detector_long = Angle(detector_long)
            detector_long = detector_long.deg
        #longitude in range (-180,180]

        self.detector_location = EarthLocation(lat=detector_lat*u.deg, lon=detector_long*u.deg, height=detector_height_m*u.m)


    def parse_date_string(
            self,
            date_str,
            utc_offset_hr = 0
            ):
        
        # Define the format of the input date string with seconds
        date_format = "%B %d, %Y, %H:%M:%S"
        # Try parsing the date string with seconds
        try:
            datetime_obj = datetime.strptime(date_str, date_format)
            return datetime_obj
        
        except ValueError:
            # If parsing with seconds fails, try parsing without seconds
            date_format_no_seconds = "%B %d, %Y, %H:%M"
            try:
                datetime_obj = datetime.strptime(date_str, date_format_no_seconds)
                return datetime_obj
            except ValueError:
                raise ValueError("Invalid date format. Please provide a date in the form 'July 15, 2023, 14:30' or 'July 15, 2023, 14:30:25'.")
                
        # Adjust datetime_obj with the UTC offset
        datetime_obj -= timedelta(hours = utc_offset_hr)

    def get_coszen_and_azimuth(
            self, 
            ra, #declination in #deg
            dec,  #right ascension in #deg
            date_str, 
            utc_offset_hr=0):
        '''
        Get the cosine of the zenith angle and the azimuth corresponding to the given 
        declination and right ascension (at the specified date/time). Depends on the detector location
        '''

        # Create observation time object
        date_obj = self.parse_date_string(date_str, utc_offset_hr)
        #Time creates time object in specific format 
        
        # Create neutrino direction
        #TODO should this be the opposite direction (e.g. travel direction vs origin direction?)
        nu_dir = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        #SkyCoord object that is per default in the icrs frame
        
        # Convert to local coordinate system (zenith, azimuth)
        #Coordinate transformation to the horizontal coordinate system. Azimuth is oriented East of North (i.e., N=0, E=90 degrees) 
        #Altitutde is the angle above horizon
        frame = AltAz(obstime=date_obj, location=self.detector_location)
        nu_dir_alt_az = nu_dir.transform_to(frame)

        # Get zenith angle and convert to cos(zen)
        coszen = np.cos( nu_dir_alt_az.zen.to(u.rad).value )
        
        # Get azimuthal angle
        azimuth = nu_dir_alt_az.az.to(u.deg).value
        
        return coszen, azimuth
    
    
    def get_right_ascension_and_declination(
            self,
            coszen,
            azimuth,
            date_str,
            utc_offset_hr = 0):
        '''
        Get the right ascension and declination corresponding to the given azimuth and zenith angle (cosine of zenith angle)
        at the specified date/time. Depends on the detector location.

        Parameters:
        coszen (array-like): Cosine of the zenith angle(s) of the neutrinos.
        azimuth (array-like): Azimuth angle(s) of the neutrinos.
        date_str (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        utc_offset_hr (float): UTC offset in hours for the specified location.

        Returns:
        tuple: A tuple containing two arrays with the right ascension and declination in degrees.
        '''

        # Create observation time object
        date_obj = self.parse_date_string(date_str,utc_offset_hr)

        # Create neutrino direction in AltAz frame
        nu_dir_alt_az = AltAz(alt=np.arccos(coszen)*u.rad, az=azimuth*u.deg, obstime=date_obj, location=self.detector_location)

        # Transform back to ICRS frame (right ascension, declination)
        nu_dir_icrs = nu_dir_alt_az.transform_to('icrs')

        # Get right ascension and declination in degrees
        right_ascension = nu_dir_icrs.ra.degree
        declination = nu_dir_icrs.dec.degree

        return right_ascension, declination
    
    def path_length(
            self,
            coszen, #array-like
            #Height at which neutrino gets produced (in km above the surface [at 0m altitude] )
            prodHeight,
            #Depth at which the detector lies (in km below the surface [at 0m altitude]) 
            detectorDepth,
            ):
    
        '''
        Calculate the path length of neutrinos from the production height to the detector.

        Parameters:
        coszen (array-like): Cosine of the zenith angle(s) of the neutrinos.
        prodHeight (float): Height at which the neutrinos are produced above the Earth's surface (in km).
        detectorDepth (float): Depth at which the detector lies below the Earth's surface (in km).

        Returns:
        numpy.ndarray: An array containing the path lengths of neutrinos in the same shape as coszen.       
        '''
        
        # Approximate Earth's radius in km
        rEarth = 6371
        
        #Create array of the same form as coszen variable to store pathlengths
        path_lengths = np.zeros_like(coszen)
        
        #Calculate pathlength
        expression = (rEarth - detectorDepth)**2 * (-1 + coszen**2) + (rEarth + prodHeight)**2
        # Check for non-negative values
        mask = expression >= 0  
        path_lengths = -rEarth * coszen + np.sqrt(expression[mask])
        
        return path_lengths
