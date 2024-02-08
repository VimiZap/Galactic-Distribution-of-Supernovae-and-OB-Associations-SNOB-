import numpy as np
import matplotlib.pyplot as plt
from pyvo import registry
import obs_utilities as ut


def get_catalogue_data(catalogue, table_name, column_names):
    """
    Get data from a VizieR catalogue using the pyvo library
    Args:
        catalogue: str. The name of the VizieR catalogue. E.g: "J/AJ/156/84"
        table_name: str. The name of the table in the catalogue. E.g: "table3"
        column_names: list of str. The names of the columns to retrieve from the table. E.g: ["Nstars", "Age-CMD", "RAJ2000", "DEJ2000", "Plx"]. MUST be double quoted if the column name contains special characters or spaces.
    """
    catalogue_ivoid = f"ivo://CDS.VizieR/{catalogue}"
    voresource = registry.search(ivoid=catalogue_ivoid)[0]
    table = catalogue + "/" + table_name
    quoted_column_names = [f'"{name}"' for name in column_names]  # Add double quotes around each column name
    adql_query = f"""
    SELECT {', '.join(quoted_column_names)}
    FROM "{table}"
    """
    tap_records = voresource.get_service("tap").run_sync(
        adql_query,
    )
    return tap_records

ORION_CATALOGUE = "J/AJ/156/84"
ORION_TABLE = "table3"
ORION_COLUMNS = ["Nstars", "Age-CMD", "RAJ2000", "DEJ2000", "Plx"]
tap_records = get_catalogue_data(ORION_CATALOGUE, ORION_TABLE, ORION_COLUMNS)
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


