#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 01:09:33 2017

@author: jmilli
"""

import astropy.units as u
import numpy as np
import os
from astropy.io import ascii
from scipy.interpolate import interp1d
__version__ = '1.0'

class EvolutionaryModel():
    """ 
    Object that parses the table provided by France Allard (http://perso.ens-lyon.fr/france.allard/
    click on a model and you'll find ascii tables in the folder ISOCHRONES) 
    to be able to derive the property of a planet given another property and the
    age of the star. Typically one would know the luminosity of the planet and 
    one would like to derive the mass assuming a certain age.
    It can also be used reversely to find the luminosity given its mass.
    Several examples of how the tool can be used are given in the __main__ below.

    Class attributes:
        - _path: the absolute path where the ascii files are stored.
    Attributes:
        -listOfDicos: a list of dictionaries. Each dictionary corresponds to an age
                        and has different entries for the different columns of the 
                        evolutionary model file
        - age: a list of tabulated ages in Gyrs
        - Mstar: mass of the star in stellar mass
        - Mplanet: mass of the planet in Jupiter mass
        - distance: the distance of the star in parsec
    Functions:
        - _interpolate_age
        - interpolate_property
    """ 
    
    #This path has to be adjusted to the location where the evolutionary models 
    # are stored in txt file
    _path = os.path.dirname(os.path.abspath(__file__))    
#    _path = '/Users/jmilli/Dropbox/lib_py/evolutionary_model'
    
    def __init__(self,model='AMES-Cond',ins='SPHERE',distance=None,**kwargs):
        """
        Constructor of the EvolutionaryModel object. It reads the ascii table 
        
        Input:
            - model: to be chosen between 'AMES-Cond', 'BT-Settl' or 'AMES-dusty'
            or the name of a text file tabulated in the same way as the other
            files (for instance 'model.AMES-Cond-2000.M-0.0.SPHERE.Vega.txt').
            - ins: instrument ('SPHERE','NaCo','2MASS'... depending on the 
                    instrument models downloaded in _path from F. Allard's website)
            - distance: (optional) distance of the star in parsec. (if 
             you provide the distance, then the apparent magnitude for each filter 
             is available through the key '<filter_name>_apparent')
            - the magnitude of the star can be added in a additional keyword 
                (for instance B_Ks=7 if B_Ks is a filter given in the file. In this 
                case the planet constrast is available through the key 'B_Ks_contrast')
        """
        if model == 'AMES-Cond':
            filename = 'model.AMES-Cond-2000.M-0.0.{0:s}.Vega.txt'.format(ins)
        elif model == 'AMES-dusty':
            filename = 'model.AMES-dusty.M-0.0.{0:s}.Vega.txt'.format(ins)
        elif model == 'BT-Settl':
            filename = 'model.BT-Settl.M-0.0.{0:s}.Vega.txt'.format(ins)
        else:
            filename = model        
        file = os.path.join(self._path,filename)
        try:
            with open(file, "r") as myfile:
                lines = myfile.readlines()
            nb_lines = len(lines)
            print('Reading {0:s} ({1:d} lines)'.format(filename,nb_lines))
        except Exception as e:
            print('Problem while reading {0:s}'.format(filename))
            print(e)
            return

        self.distance = distance
        self.listOfDicos=[]
        line_index_age_bin = []
        for k,v in kwargs.items():
            print('Star magnitude in {0:s}: {1:.3f}'.format(k,v))
        self.age = []
        for i,line in enumerate(lines):
            if line.replace(' ','').startswith('t(Gyr)='):            
                line_index_age_bin.append(i)
                self.age.append(float(line[line.index('=')+1:].strip()))
        for i,index in enumerate(line_index_age_bin):
            finalColumnName = lines[index+2].replace('M/MsTeff(K)','M/Ms	Teff(K)').split() # this is a fix because the first 2 column names are without white spaces
            if i!=len(line_index_age_bin)-1:
                table_tmp = ascii.read(lines[index+4:line_index_age_bin[i+1]-4],format='no_header')
            else:
                table_tmp = ascii.read(lines[index+4:-1],format='no_header')
            if len(finalColumnName)!=len(table_tmp.columns):
                raise Exception('Unable to match the column names and data')
            else:
                origColumnNames = table_tmp.colnames
                for i,origColumnName in enumerate(origColumnNames):
                    table_tmp.rename_column(origColumnName,finalColumnName[i])
                dico_tmp = dict(table_tmp)
                if 'M/Ms' in dico_tmp.keys():
                    dico_tmp['Mplanet'] = (np.asarray(dico_tmp['M/Ms'])*u.Msun/u.Mjupiter).to(u.dimensionless_unscaled)
                if self.distance != None:
                    for key in finalColumnName[6:]: # we skip the first columns that are not magnotude
                        #dico_tmp[key] is an absolute magnitude
                        dico_tmp[key+'_apparent'] = np.asarray(dico_tmp[key])+ 5*(np.log10(self.distance)-1)
                        if key in kwargs.keys():
                            dico_tmp[key+'_contrast'] = dico_tmp[key+'_apparent'] - kwargs[key]
            self.listOfDicos.append(dico_tmp)
    
    def _find_lower_upper_index(self,value,array):
        """
        Utility function that finds the lower and upper index of an array that 
        surrounds a given value
        The value must within the range of the array.
        """
        array = sorted(array)
        if value < array[0] or value > array[-1]:
            raise ValueError('The value is out of range: {0:.3f} is outside [{1:.3f};{2:.3f}]'.format(value,array[0],array[-1]))
        for i,value_tmp in enumerate(array[1:]):
            if value_tmp>=value:
                return i,i+1

    def _interpolate_age(self,age_value,prop1,prop2):
        """
        Returns 2 arrays for 2 different properties interpolated linearily 
        for the requested age
        Input:
            - age_value: age in Gyr
            - prop1 and prop2: string corresponding to the column name requested
            (generally 'M/Ms','Teff(K)','L/Ls','lg(g)','R(Gcm','D','Li','J')
        Restriction: this function does not always work if the prop1 and prop2
        arrays for the 2 different ages (upper abd below) have a different dimension
        """
        indices = self._find_lower_upper_index(age_value,self.age)
        age_before,age_after = self.age[indices[0]],self.age[indices[1]]
        print('Interpolating {0:.3f} linearily between {1:.3f} Gyrs and {2:.3f} Gyrs'.format(age_value,age_before,age_after))
        prop1_below = self.listOfDicos[indices[0]][prop1]
        prop1_above = self.listOfDicos[indices[1]][prop1]
        prop1_array = prop1_below + (prop1_above-prop1_below)/(age_after-age_before)*(age_value-age_before)   
        prop2_below = self.listOfDicos[indices[0]][prop2]
        prop2_above = self.listOfDicos[indices[1]][prop2]
        prop2_array = prop2_below + (prop2_above-prop2_below)/(age_after-age_before)*(age_value-age_before)   
        return prop1_array,prop2_array

    def interpolate_property(self,age_value,prop1_value,prop1,prop2):
        """
        For a given age and a value for the property 1, it interpolates linearily 
        to return the corresponding value for property 2.
        Input:
            - age_value: age in Gyr
            - prop1_value: the value of property 1 for which we need to interpolate
            - prop1 and prop2: strings corresponding to the column name requested
            (generally 'M/Ms','Teff(K)','L/Ls','lg(g)','R(Gcm','D','Li','J')
        """
#        prop1_array,prop2_array = self._interpolate_age(age_value,prop1,prop2)
#        interp_function = interp1d(prop1_array,prop2_array,bounds_error=True)
#        try:
#            prop2_value = float(interp_function(prop1_value))
#        except ValueError as e:
#            print('ValueError: {0:s}'.format(str(e)))
#            print('The property {0:s} of {1:.5f} is out of range'.format(prop1,prop1_value))
#            print('Returning NaN')
#            return np.nan
#        print('For an age of {0:.4f} Gyr and {1:s}={2:.3f}, the model gives {3:s}={4:.3f}'.format(
#                age_value,prop1,prop1_value,prop2,prop2_value))
#        return prop2_value

        indices = self._find_lower_upper_index(age_value,self.age)
        age_before,age_after = self.age[indices[0]],self.age[indices[1]]
        print('Interpolating {0:.3f} linearily between {1:.3f} Gyrs and {2:.3f} Gyrs'.format(age_value,age_before,age_after))
        prop1_below = self.listOfDicos[indices[0]][prop1]
        prop1_above = self.listOfDicos[indices[1]][prop1]
        prop2_below = self.listOfDicos[indices[0]][prop2]
        prop2_above = self.listOfDicos[indices[1]][prop2]
        interp_function_below = interp1d(prop1_below,prop2_below,bounds_error=True)
        interp_function_above = interp1d(prop1_above,prop2_above,bounds_error=True)
        try:
            prop2_below_value = interp_function_below(prop1_value)
            prop2_above_value = interp_function_above(prop1_value)
        except ValueError as e:
            print('ValueError: {0:s}'.format(str(e)))
            print('The property {0:s} of {1:.5f} is out of range'.format(prop1,prop1_value))
            print('Returning NaN')
            return np.nan
        prop2_value = prop2_below_value + (prop2_above_value-prop2_below_value)/(age_after-age_before)*(age_value-age_before)  
        return prop2_value
    
if __name__=='__main__':
    #you create the object. In this case, we specify the stellar mass (to get the 
    # planet mass) but also the distance (to get the absolute mag) and the B_Ks
    # magnitude (to get the contrast)

#    model = EvolutionaryModel(model='BT-Settl',ins='2MASS')
#    mag_abs_J_pl = model.interpolate_property(0.130,0.1,'M/Ms','J')    
#    
#    model = EvolutionaryModel(distance=4,B_Ks=4)
#    apparent_mag_Ks = model.interpolate_property(0.01,5,'Mplanet','B_Ks_apparent')
#    contrast_Ks = model.interpolate_property(0.01,5,'Mplanet','B_Ks_contrast')
#    contrast_Ks = model.interpolate_property(0.01,5,'B_Ks_contrast','Mplanet')
#
#    # if for some reasons you want the full arrays:
#    mass_ratio_array,absolute_mag_Ks = model._interpolate_age(0.01,'M/Ms','B_Ks')
#
#    model = EvolutionaryModel(model='AMES-Cond',ins='2MASS',distance=10,K=5)
#    planet_mass = model.interpolate_property(0.9,5,'K_contrast','Mplanet')
#
#    # Second set of test tailored for HD206893:
#    model = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',distance=38.34,B_Ks=5.593)
##    planet_mass = model.interpolate_property(0.2,16.79,'B_H_apparent','Mplanet')
#    contrast_H = model.interpolate_property(0.05,15,'Mplanet','B_Ks_contrast') 
#    print(np.power(10,-contrast_H/2.5)) #1.2e-4
#    planet_radius_Gcm = model.interpolate_property(0.2,16.79,'B_H_apparent','R(Gcm')
#    planet_Teff = model.interpolate_property(0.2,16.79,'B_H_apparent','Teff(K)')
#    planet_logg = model.interpolate_property(0.2,16.79,'B_H_apparent','lg(g)')
#    planet_radius_Rjup = planet_radius_Gcm*0.01 * 1.e9 / (u.R_jup.to(u.meter))
#    print(planet_radius_Rjup)
# 
#    model = EvolutionaryModel(model='AMES-Cond',ins='NaCo',distance=38.34,Lprime=5.52)
#    planet_mass = model.interpolate_property(0.2,13.43,'Lprime_apparent','Mplanet')
#    planet_Teff = model.interpolate_property(0.2,16.79,'Lprime_apparent','Teff(K)')
#    planet_radius_Gcm = model.interpolate_property(0.2,16.79,'Lprime_apparent','R(Gcm')
#    planet_radius_Rjup = planet_radius_Gcm*0.01 * 1.e9 / (u.R_jup.to(u.meter))
#    print(planet_radius_Rjup)
#    planet_logg = model.interpolate_property(0.2,16.79,'Lprime_apparent','lg(g)')
#    planet_radius_Rjup = planet_radius_Gcm*0.01 * 1.e9 / (u.R_jup.to(u.meter))
#
    # For beta Pic with SPHERE
#    model_bPic_NaCo = EvolutionaryModel(model='AMES-Cond',ins='NaCo',distance=19.44,Lprime=5.52)
#    age = 0.023
#    contrast_Lp = model_bPic_NaCo.interpolate_property(age,10,'Mplanet','Lprime_contrast') #7.75
#    model_bPic_SPHERE = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',Mstar=1.64,distance=19.44,B_H=3.5,B_Ks=3.5)
#    contrast_H = model_bPic_SPHERE.interpolate_property(age,10,'Mplanet','B_H_contrast') #12.2
#    contrast_Ks = model_bPic_SPHERE.interpolate_property(age,10,'Mplanet','B_Ks_contrast') #11.65
#    print(np.power(10,-contrast_H/2.5))
#    print(np.power(10,-contrast_Ks/2.5))
#    contrast_Ks_measured = 11.2-2
#    print(np.power(10,-contrast_Ks_measured/2.5))
#    
#    model = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',distance=100.0,B_H=8.4)
#    planet_mass = model.interpolate_property(0.06,np.array([1,2,3]),'B_H_contrast','Mplanet')
#    print(planet_mass)
#
#    model = EvolutionaryModel(model='BT-Settl',ins='2MASS',distance=1./0.026,H=5.9)
#    planet_mass = model.interpolate_property(0.06,7.25,'H_contrast','Mplanet')
#    print(planet_mass)
#    
#    model = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',distance=1/0.00984,D_K1=6.65)
#    planet_mass = model.interpolate_property(0.01,6.35,'D_K1_contrast','Mplanet')
#    print(planet_mass)
#    
#    model = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',distance=1/0.014,B_H=5.664)
#    for age in np.array([0.01,0.045,0.1,0.5]):
#        planet_mass = model.interpolate_property(age,10.6,'B_H_contrast','Mplanet')
#        print(planet_mass)

#    # For GJ3998
#    contrast=6.8e-6
#    delta_mag = -2.5*np.log10(contrast)
#    mag_star = 7.02
#    separation = 4.9#arcsec
#    star_dist = 1/0.055#pc
#    separation_au = separation*star_dist
#    mag_companion = mag_star+delta_mag
#    model = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',distance=1/0.055,B_H=7.02)
#    for age in np.array([0.1,0.5,1,5]):
#        planet_mass = model.interpolate_property(age,delta_mag,'B_H_contrast','Mplanet')
#        print(planet_mass)

#    # For AU Mic
#    contrast=0.00005
#    delta_mag = -2.5*np.log10(contrast)
##    delta_mag = 7.5
#    mag_star = 4.8
#    star_dist = 1./0.1028#pc
#    age = 0.023
#    model = EvolutionaryModel(model='AMES-Cond',ins='SPHERE',distance=star_dist,B_Ks=mag_star)
#    planet_mass = model.interpolate_property(age,delta_mag,'B_Ks_contrast','Mplanet')
#    print(planet_mass)
    
    # For HD95086
    model = EvolutionaryModel(model='AMES-Cond',ins='NaCo',distance=90.4,Lprime=6.70) 
    age=0.017
    contrast_lp=9.79
    planet_mass = model.interpolate_property(age,contrast_lp,'Lprime_contrast','Mplanet')
    print(planet_mass)
    
