import starfile
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

def subsample_star_file(star: pd.DataFrame, 
                        fractions: None | np.ndarray = None,
                        num_replicates: int = 5):
    if fractions is None:
        fractions = np.arange(0.05, 1.0, 0.05)



    subsampled_stars = []
    subsampled_fractions = []
    for fraction in fractions:
        for rep in range(num_replicates):
            frac_sample = star.sample(frac=fraction, random_state=42 + rep).reset_index(drop=True)
            subsampled_stars.append(frac_sample)
            subsampled_fractions.append(fraction)

    return subsampled_stars, np.array(subsampled_fractions)

def create_star_with_voids(star: pd.DataFrame, void_radius: int = 100, num_trials: int = 20):

    void_centres = star.sample(n=num_trials, random_state=42)[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values

    stars = []
    num_points = []
    for void_centre in void_centres:
        dist_mat = np.linalg.norm(void_centre - star[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values, axis=1)
        star_void = star.loc[dist_mat > void_radius, :]
        stars.append(star_void)
        num_points.append(len(star_void))

    return stars, np.asarray(num_points), void_centres



def approx_equal(a, b, tol=1e-6):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return np.abs(a - b) < tol