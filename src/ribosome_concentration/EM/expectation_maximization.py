import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import factorial
import einops 
import matplotlib.pyplot as plt 
import starfile
import pandas as pd
from tqdm import tqdm
import napari
from ribosome_concentration.input.input import read_tomogram, rescale_star_coordinates, read_star_file


rng = np.random.default_rng(42)

def create_synthetic_tomogram(reconstruction_dims=(30, 58, 41), slab_height=(10, 58, 41), probability=1e-3):
    synthetic_tomogram = np.zeros(reconstruction_dims)
    z_start = reconstruction_dims[0] // 2 - slab_height[0] // 2
    z_end = z_start + slab_height[0]
    synthetic_tomogram[z_start:z_end, :, :] = rng.choice((0, 1), size=slab_height, p=(1-probability, probability))
    synthetic_points = np.argwhere(synthetic_tomogram == 1)
    return synthetic_tomogram, z_start, z_end, synthetic_points


def sample_regular_grid(tomogram, R):
    z = np.arange(R, tomogram.shape[0]-R+1, 2 * R, dtype=int)
    y = np.arange(R, tomogram.shape[1]-R+1, 2 * R, dtype=int)
    x = np.arange(R, tomogram.shape[2]-R+1, 2 * R, dtype=int)
#    print(f"Sampling regular grid with points at z values {np.unique(z)}.")
    grid = np.meshgrid(z, y, x, indexing='ij')
    return einops.rearrange(np.asarray(grid).T, 'i j k l -> (i j k) l')


def expectation_step(x_i, T_ji):
    lambda_j = T_ji @ x_i / np.sum(T_ji, axis=1)
    tau_j = np.sum(T_ji, axis=1) / T_ji.shape[1]

    return lambda_j, tau_j


def maximization_step(x_i, lambda_j, tau_j):
    T_ji = np.zeros((len(lambda_j), len(x_i)))
    for j in range(len(lambda_j)):
        T_ji[j, :] = tau_j[j] * np.exp(-lambda_j[j]) * lambda_j[j] ** x_i # / factorial(x_i) (as this can be factored out with numerical advantages)
    T_ji = T_ji / np.sum(T_ji, axis=0, keepdims=True)
    return T_ji


def calculate_expected_likelihood(x_i, T_ji, lambda_j, tau_j):
    val = 0
    for i in range(T_ji.shape[1]):
        for j in range(T_ji.shape[0]):
            val += T_ji[j, i] * (np.log(tau_j[j]) - lambda_j[j] + x_i[i] * np.log(lambda_j[j])) # - np.log(factorial(x_i[i])))

    return val


def em_algorithm(x_i, lambda_j=np.array([1, 0.1]), tau_j=np.array([0.3, 0.7]), max_iterations=1e3, tol=1e-2):

    expected_likelihood = []
    for iteration in range(int(max_iterations)):
        T_ji = maximization_step(x_i, lambda_j, tau_j)
        lambda_j, tau_j = expectation_step(x_i, T_ji)
        expected_likelihood.append(calculate_expected_likelihood(x_i, T_ji, lambda_j, tau_j))
#        print(f"Iteration {iteration}: Expected Likelihood = {expected_likelihood[-1]}")

        if iteration > 0 and np.linalg.norm(expected_likelihood[-1] - expected_likelihood[-2]) < tol:
            break

    res = {
        "lambda" : lambda_j,
        "tau" : tau_j,
        "mixture_probability" : T_ji,
        "expected_likelihood": expected_likelihood
    }

    return res


def em_analyse_tomogram(tomogram, star, tomogram_apx, star_file_apx, poisson_radius):
    mat = star[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values.astype(int)
    rescaled_mat = rescale_star_coordinates(mat, tomogram_apx, star_file_apx).T[:, ::-1]
    reg_grid = sample_regular_grid(tomogram, R=poisson_radius)
    dist_mat = distance_matrix(reg_grid, rescaled_mat)
    poisson_data = np.sum(dist_mat < poisson_radius, axis=1)
    res_tomo = em_algorithm(poisson_data, lambda_j=np.array([1, 0.1]), tau_j=np.array([0.3, 0.7]), max_iterations=1e3, tol=1e-2)

    return rescaled_mat, reg_grid, poisson_data, res_tomo
        
def batch_em(tomogram_dir, star_file, tomogram_apx, star_file_apx, poisson_radius):
    star = read_star_file(star_file)

    tomograms = []
    rescaled_coordinates = []
    reg_grids = []
    poisson_data_all = []
    em_results_per_tomogram = []
    for name in tqdm(star["rlnMicrographName"].unique()):
        star_filt = star.loc[star.rlnMicrographName == name]
        tomogram_path = f"{tomogram_dir}/{name}.mrc"

        tomogram, _ = read_tomogram(tomogram_path)
        if tomogram is None:
            continue

        if tomogram is not None:
            print(tomogram_path)

        tomograms.append(tomogram)

        rescaled_mat, reg_grid, poisson_data, res_tomo = em_analyse_tomogram(tomogram, star_filt, tomogram_apx, star_file_apx, poisson_radius)

        rescaled_coordinates.append(rescaled_mat)
        reg_grids.append(reg_grid)
        poisson_data_all.append(poisson_data)
        em_results_per_tomogram.append(res_tomo)

    poisson_data_concatenated = np.concatenate(poisson_data_all)

    res_all = em_algorithm(poisson_data_concatenated, lambda_j=np.array([1, 0.1]), tau_j=np.array([0.3, 0.7]), max_iterations=1e3, tol=1e-2)


    lambda_values = np.asarray([res["lambda"] for res in em_results_per_tomogram])
    lambda_value_all = np.asarray(res_all["lambda"])

    molar_concentrations = convert_to_molar_concentration(lambda_values, poisson_radius, tomogram_apx=tomogram_apx)
    molar_concentration_all = convert_to_molar_concentration(lambda_value_all, poisson_radius, tomogram_apx=tomogram_apx)
    return tomograms, rescaled_coordinates, reg_grids, poisson_data_all, em_results_per_tomogram, res_all, molar_concentrations, molar_concentration_all

def convert_to_molar_concentration(lamb, R, tomogram_apx, avogadro_number=6.022e23):
    sampling_volume_m3 = (4/3) * np.pi * (R * tomogram_apx * 1e-10) ** 3
    sampling_volume_L = sampling_volume_m3 * 1e3  # convert m^3 to L
    molar_concentration = lamb / (sampling_volume_L * avogadro_number)  # mol/L
    return molar_concentration


def visualize_em_results_per_tomogram(tomogram, rescaled_mat, reg_grid, res, R):
    viewer = napari.Viewer()
    viewer.add_image(tomogram, name='tomogram')
    viewer.add_points(rescaled_mat, size=8, name='particles', out_of_slice_display=True)
    viewer.add_points(reg_grid, size=2 * R, 
                    properties={'cytosol_probability': res["mixture_probability"][0, :]},
                    name='regular_grid', 
                    face_color='cytosol_probability',
                    face_colormap='BuGn',
                    opacity=0.5,
                    out_of_slice_display=True)
    napari.run()


def pois_dist(lam, max=100):
    arr = np.zeros((max,))
    for k in range(max):
        arr[k] = lam**k * np.exp(-lam)/factorial(k)
    return np.arange(max), arr
