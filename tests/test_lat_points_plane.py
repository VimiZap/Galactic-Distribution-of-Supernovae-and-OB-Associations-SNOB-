import numpy as np
import matplotlib.pyplot as plt


def points_latitude(b_min=0.01, b_max=1, scaling=0.015, db_above_1_deg = 0.2):
    """
    Args:
        b_min: minimum angular distance from the plane
        b_max: maximum angular distance from the plane
        scaling: scaling of the exponential distribution. A value of 0.015 generates between 110 - 130 points. Larger scaling = fewer points
    Returns:
        1D array with the angular distances from the Galactic plane, and an array with all increments (db) between the points
        Arrays intended to be used for a central Rieman sum 
    """
    num_lats_above_1_deg = 4 / db_above_1_deg
    if num_lats_above_1_deg == int(num_lats_above_1_deg): # to make sure that the number of latitudinal angles between 1 and 5 degrees is an integer 
        db = np.array([b_min])
        while np.sum(db) < b_max: # to make sure that the first element shall be b_min
            db = np.append(db, np.random.exponential(scale=scaling) + b_min)
        # now the sum of db is larger than b_max, so we need to adjust for that
        diff = np.sum(db) - (b_max)
        db.sort()
        db[-1] = db[-1] - diff
        db.sort()
        latitudes = np.cumsum(db)
        # move each dr, such that they are centralized on each point. Essential for the Riemann sum
        dr_skewed = np.append(db[1:], db_above_1_deg/2)
        db = (db + dr_skewed) / 2  
        # transverse_distances done for b_min <= b <= b_max. Now we need to do b_max < b <= 5
        if latitudes[0] == b_min: # to make sure that the first element is be b_min
            latitudes = np.append(latitudes, latitudes[-1] + db_above_1_deg / 2)
            db = np.append(db, db_above_1_deg * 0.75)
            for _ in range(1, int(num_lats_above_1_deg)):
                latitudes = np.append(latitudes, latitudes[-1] + db_above_1_deg)
                db = np.append(db, db_above_1_deg)
            latitudes = np.concatenate((-latitudes[::-1], [0], latitudes))
            db = np.concatenate((db[::-1], [b_min], db))
            return latitudes, db
        else:
            print("The first element of transverse_distances is not b_min. Trying again...")
            return points_latitude(b_min, b_max, scaling, db_above_1_deg)
    else:
            raise ValueError("The number of latitudinal angles is not an integer. Please change the value of db_above_1_deg")

test_lat, db = points_latitude()
print(test_lat)
print(len(test_lat))
print(db)
print(len(db))
print("Sum of all elements in db:", np.sum(db))
lats = np.arange(0, 51, 1)
longitudes = np.arange(0, 361, 1)
radial_distances = np.arange(0, 21, 1)
print(len(radial_distances))
lats = np.tile(lats, (1, len(longitudes), len(radial_distances)))
print(lats.shape)
print(lats)
