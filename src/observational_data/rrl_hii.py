import numpy as np
import matplotlib.pyplot as plt
from pyvo import registry



# each resource in the VO has an identifier, called ivoid. For vizier catalogs, # the VO ids can be constructed like this:
CATALOGUE = 'J/ApJS/165/338'
catalogue_ivoid = f"ivo://CDS.VizieR/{CATALOGUE}"
# the actual query to the registry
voresource = registry.search(ivoid=catalogue_ivoid)[0]

# We can extract the tables from the resource
tables = voresource.get_tables()
# We can also extract the tables names for later use
tables_names = list(tables.keys())
print(tables_names)
# extract the table name of interest
table_1 = tables[tables_names[2]]
print("table_1: ", table_1)
# investigate the columns of the table
column_names = [column.name for column in table_1.columns]
print(column_names)

# Define your ADQL query
adql_query = """
SELECT column_name
FROM table_name
"""
tap_records = voresource.get_service("tap").run_sync(
    f'SELECT glat from "{tables_names[2]}"',
)
print("number of tap_records: ", len(tap_records))

print(tap_records)
glat_data = tap_records['GLAT'].data
glon_data = tap_records['GLON'].data

bin_edges = np.arange(-4, 4+0.5, 0.5)
binned_data, bin_edges = np.histogram(glat_data, bins=bin_edges)
plt.bar(bin_edges[:-1], binned_data, width=0.5, align='edge')
plt.xlabel('glat')
plt.ylabel('count')
plt.title('Histogram of glat')
plt.show()
print("Number of points in the binned data: ", np.sum(binned_data))
print("Number of points with glat < abs(1): ", len(glat_data[np.abs(glat_data) < 1]))
print("Number of points with glat > abs(4) in the original dataset:", len(glat_data[np.abs(glat_data) > 4]))
print("Number of points with exactly glat = +1 or -1: ", len(glat_data[np.abs(glat_data) == 1]))
print(glat_data[np.abs(glat_data) > 4])
