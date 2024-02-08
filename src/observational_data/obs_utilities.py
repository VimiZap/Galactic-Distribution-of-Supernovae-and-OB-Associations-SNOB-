from astropy.coordinates import SkyCoord
import astropy.units as u

""" # Example coordinates in RA and DEC (in decimal degrees)
ra_dec_coordinates = [
    (10.684, 41.269),  # RA and DEC for a sample star
    # You can add more coordinates here
]

# Convert RA and DEC to Galactic coordinates
galactic_coordinates = []

for ra, dec in ra_dec_coordinates:
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galactic = coord.galactic
    galactic_coordinates.append((galactic.l.degree, galactic.b.degree))

print(galactic_coordinates) """

def ra_dec_to_galactic(ra, dec):
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galactic = coord.galactic
    return galactic.l.degree, galactic.b.degree


def mas_to_parsec(mas):
    """
    Convert milliarcseconds to parsecs
    Args:
        mas: float. Milliarcseconds
    Returns:
        float. Parsecs
    """
    return 1 / (mas / 1000)