import numpy as np
import matplotlib.pyplot as plt
from pyvo import registry
import obs_utilities as ut




ORION_CATALOGUE = "J/AJ/156/84"
ORION_TABLE = "table3"
ORION_COLUMNS = ["Nstars", "Age-CMD", "RAJ2000", "DEJ2000", "Plx"]
tap_records = ut.get_catalogue_data(ORION_CATALOGUE, ORION_TABLE, ORION_COLUMNS)
print("Number of datapoints: ", len(tap_records))

orion_ra = tap_records['RAJ2000'].data
orion_dec = tap_records['DEJ2000'].data
orion_l, orion_b = ut.ra_dec_to_galactic(orion_ra, orion_dec)
orion_num_stars = tap_records['Nstars'].data
print("Number of stars in the Orion catalogue: ", np.sum(orion_num_stars))
# Convert parallax to distance
orion_plx = tap_records['Plx'].data
orion_distance = ut.mas_to_parsec(orion_plx)

for i in range(5):
    print(f"Glon: {orion_l[i]}, Glat: {orion_b[i]}, Parsec: {orion_distance[i]}")


