from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import norm
import ribosome_concentration.EM.expectation_maximization as em
import ribosome_concentration.input.input as inp
import os 
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_array
from ribosome_concentration.input.input import read_tomogram, rescale_star_coordinates, read_star_file
import einops 
import napari

def convert_to_coo_format(coordinates, indices, dat, shape, dilation_factor = 1):
    if indices is None:
        indices = np.arange(coordinates.shape[0])
    coo_x = coordinates[indices][:, 0] * dilation_factor
    coo_y = coordinates[indices][:, 1] * dilation_factor
    coo_z = coordinates[indices][:, 2] * dilation_factor
    coo_data = dat
    shape = np.asarray(shape, dtype=int)
    spatial_occupancy_array = coo_array((coo_data, (coo_x, coo_y, coo_z)), shape=np.asarray(shape * dilation_factor, dtype=int))

    return spatial_occupancy_array

class Tomogram:
    avogadro_number = 6.022e23
    def __init__(self, data, star_coordinates, tomogram_apx, star_file_apx):
        self.tomogram_apx = tomogram_apx
        self.star_file_apx = star_file_apx
        self.particles = rescale_star_coordinates(star_coordinates, tomogram_apx, star_file_apx).T[:, ::-1]
        self.particles_coo = convert_to_coo_format(self.particles, np.arange(self.particles.shape[0]), np.ones(self.particles.shape[0]), data.shape)
        self.particles_dense = self.particles_coo.todense()
        self.data = data
        self.shape = data.shape
        self.compartment_counts = None
        self.compartment_stdevs = None
        self.forbidden_region = np.zeros(self.shape, dtype=bool)
        self.forbidden_labels = []


    def compartmentalize(self, compartment_size, centre_separation=None, x_separation=None, y_separation=None, z_separation=None):
        if centre_separation is None:
            centre_separation = compartment_size
        if x_separation is None:
            x_separation = centre_separation
        if y_separation is None:
            y_separation = centre_separation
        if z_separation is None:
            z_separation = centre_separation
        
        self.compartment_centres = get_compartment_centres(self.shape, x_separation=x_separation, y_separation=y_separation, z_separation=z_separation)
        self.compartments = [Compartment(centre_coordinate=compartment_central_coordinate, size=compartment_size, tomogram=self) for compartment_central_coordinate in self.compartment_centres]

    def obtain_compartment_stdevs(self):
        self.compartment_stdevs = []
        for compartment in self.compartments:
            self.compartment_stdevs.append(compartment.calculate_stdev())

    def obtain_compartment_counts(self):
        self.compartment_counts = []
        for compartment in self.compartments:
            self.compartment_counts.append(compartment.count_particles())

    def find_forbidden_compartments(self, mask, threshold: float = 1e-3, add=True, label=None):
        self.forbidden_labels.append(label)
        for compartment in self.compartments:
            to_forbid = compartment.find_overlap(mask) > threshold
            if to_forbid:
                 compartment.forbidden_dict[label] = True
                 compartment.is_forbidden = True
            else:
                compartment.forbidden_dict[label] = False
                if add is False:
                    compartment.is_forbidden = False


    def define_void_compartments(self, mixture_probability_threshold=0.5, plot=False):
        mu_0_initial, mu_1_initial, sigma_initial = self._define_initial_em_parameters(self.compartment_stdevs)
        self.em_res = em_algorithm(self.compartment_stdevs, 
                     mu_j=np.array([mu_0_initial, mu_1_initial]), 
                     sigma_j=np.array([sigma_initial, sigma_initial]), tau_j=np.array([0.5, 0.5]), 
                     max_iterations=1e3, tol=1e-2)
        
        if plot:
            self._plot_em_result(self.em_res)
        

        self.void_compartments = self.em_res['mixture_probability'][0] > mixture_probability_threshold


    def find_void_volume(self, 
                         plot=False, 
                         mixture_probability_threshold=0.5, 
                         proportion_compartments_threshold=1e-3,
                         proportion_volume_threshold=0.20):

        self.define_void_compartments(mixture_probability_threshold=mixture_probability_threshold, plot=plot)

        void_mask = np.zeros(self.shape, dtype=float)
        num_compartment_mask = np.zeros(self.shape, dtype=float)
        for is_void, compartment in zip(self.void_compartments, self.compartments):
            void_mask[compartment.region] += is_void
            num_compartment_mask[compartment.region] += 1
        num_compartment_mask[num_compartment_mask == 0] = 1

        void_mask /=  num_compartment_mask

        self.void_mask = void_mask
        self.void_mask_thresholded = void_mask > proportion_compartments_threshold

        self.find_forbidden_compartments(self.void_mask_thresholded, threshold=proportion_volume_threshold, label='void')
        

    
    def _define_initial_em_parameters(self, arr):
        mu_0_initial = np.quantile(arr, 0.4)
        mu_1_initial = np.quantile(arr, 0.6)
        sigma_initial = np.std(arr)
        return mu_0_initial, mu_1_initial, sigma_initial
    
    def _plot_em_result(self, em_res):


        n1 = norm(loc=em_res['mu'][0], scale=em_res['sigma'][0])
        n2 = norm(loc=em_res['mu'][1], scale=em_res['sigma'][1])

        plot_vals = np.arange(0, np.max(self.compartment_stdevs), 0.0001)
        n1_pdf = n1.pdf(plot_vals)
        n2_pdf = n2.pdf(plot_vals)

        plt.plot(plot_vals, n1_pdf * em_res['tau'][0])
        plt.plot(plot_vals, n2_pdf * em_res['tau'][1])
        plt.hist(self.compartment_stdevs, bins=50, density=True)
        plt.show()

    def forbid_empty_compartments(self):
        self.forbidden_labels.append('empty')
        for compartment in self.compartments:
            if compartment.is_empty:
                compartment.is_forbidden = True
                compartment.forbidden_dict['empty'] = True
            else:
                compartment.forbidden_dict['empty'] = False

    def reset_forbidden_compartments(self):
        for compartment in self.compartments:
            compartment.is_forbidden = False
            compartment.forbidden_dict = {}
    

    def calculate_ribosome_concentration(self):
        allowed_mask = np.zeros(self.shape, dtype=bool)
        for compartment in self.compartments:
            allowed_mask[compartment.region] = not compartment.is_forbidden

        voxel_volume_nm3 = (self.tomogram_apx * 0.1) ** 3
        voxel_volume_L = voxel_volume_nm3 * 1e-24

        self.number_of_allowed_voxels = np.sum(allowed_mask)
        self.number_of_ribosomes_in_allowed_volume = np.sum(allowed_mask * self.particles_dense)
        self.allowed_volume_L = self.number_of_allowed_voxels * voxel_volume_L
        
        self.ribosome_concentration_molar = self.number_of_ribosomes_in_allowed_volume / self.allowed_volume_L / Tomogram.avogadro_number # in Molar

    def get_forbidden_masks(self):
        forbidden_mask = np.zeros(self.shape, dtype=bool)
        for compartment in self.compartments:
            if compartment.is_forbidden:
                forbidden_mask[compartment.region] = True

        labelled_masks = {}
        for label in self.forbidden_labels:
            labelled_masks[label] = np.zeros(self.shape, dtype=bool)
            for compartment in self.compartments:
                if label in compartment.forbidden_dict and compartment.forbidden_dict[label]:
                    labelled_masks[label][compartment.region] = True 

        return forbidden_mask, labelled_masks
    

    def view_forbidden_masks(self):
        viewer = napari.Viewer()
        forbidden_mask, labelled_masks = self.get_forbidden_masks()
        viewer.add_image(self.data, name='tomogram')
        viewer.add_points(self.particles, name='particles', out_of_slice_display=True)
        for label, mask in labelled_masks.items():
            viewer.add_image(mask, name=f'forbidden_mask_{label}', colormap='red', blending='additive', opacity=0.3)
        viewer.add_image(forbidden_mask, name='forbidden_mask', colormap='red', blending='additive', opacity=0.3)

                         



def get_compartment_centres(array_shape, centre_separation=None, x_separation=None, y_separation=None, z_separation=None, centre_offset=None):
    if x_separation is None:
        x_separation = centre_separation
    if y_separation is None:
        y_separation = centre_separation
    if z_separation is None:
        z_separation = centre_separation

    centre_separation = np.array([z_separation, y_separation, x_separation])


    compartment_array_shape = np.asarray(array_shape) // np.array([z_separation, y_separation, x_separation])

    if centre_offset is not None and isinstance(centre_offset, (list, tuple, np.ndarray)):
        centre_offset = np.array(centre_offset)
    if centre_offset is not None and isinstance(centre_offset, int):
        centre_offset = np.array([centre_offset] * len(array_shape))
    if centre_offset is None:
        centre_offset = centre_separation // 2

    compartment_central_coordinates = einops.rearrange(
        np.indices(compartment_array_shape) * centre_separation[:, None, None, None], 
        'd z y x -> (z y x) d') + centre_offset
    
    return compartment_central_coordinates

class Compartment:
    def __init__(self, centre_coordinate, size, tomogram):
        self.centre_coordinate = centre_coordinate
        self.tomogram = tomogram
        self.size = size
        self.particle_count = 0
        self.is_forbidden = False
        self.region = self.obtain_region()
        region_indices = np.array([self.region[0], self.region[1], self.region[2]])
        self.points = einops.rearrange(region_indices, 'd x y z -> (x y z) d')
        self.particle_count = 0
        self.is_forbidden = False
        self.forbidden_dict = {}
        
    def obtain_region(self):
        half_size = self.size // 2
        min_bound_x = np.max([self.centre_coordinate[0] - half_size + 1, 0])
        max_bound_x = np.min([self.centre_coordinate[0] + half_size, self.tomogram.shape[0]])
        min_bound_y = np.max([self.centre_coordinate[1] - half_size + 1, 0])
        max_bound_y = np.min([self.centre_coordinate[1] + half_size, self.tomogram.shape[1]])
        min_bound_z = np.max([self.centre_coordinate[2] - half_size + 1, 0])
        max_bound_z = np.min([self.centre_coordinate[2] + half_size, self.tomogram.shape[2]])
        res = np.meshgrid(np.arange(min_bound_x, max_bound_x), np.arange(min_bound_y, max_bound_y), np.arange(min_bound_z, max_bound_z))
        return res
    
    def calculate_stdev(self):
        region_values = self.tomogram.data[self.region]
        self.stdev = np.std(region_values)
        return self.stdev
    
    def count_particles(self):
        self.particle_count = np.sum(self.tomogram.particles_dense[self.region])
        return self.particle_count
    
    def find_overlap(self, mask ):
        overlap = mask[self.region]
        return np.mean(overlap)
    
    @property
    def is_empty(self):
        return self.particle_count == 0 
    
    



def expectation_step(x_i, T_ji):
    
    mu_j = T_ji @ x_i / np.sum(T_ji, axis=1)
    sigma_j = np.zeros(len(mu_j))
    for j in range(len(mu_j)):
        sigma_j[j] = ((np.sum((x_i - mu_j[j]) ** 2 * T_ji[j, :])) / np.sum(T_ji[j, :])) ** (1/2)
    tau_j = np.sum(T_ji, axis=1) / T_ji.shape[1]

    return mu_j, sigma_j, tau_j


def maximization_step(x_i, mu_j, sigma_j, tau_j):
    T_ji = np.zeros((len(mu_j), len(x_i)))
    for j in range(len(mu_j)):
        T_ji[j, :] = tau_j[j] / sigma_j[j] * np.exp(-(x_i - mu_j[j]) ** 2/ (2 * sigma_j[j] ** 2))
    T_ji = T_ji / np.sum(T_ji, axis=0, keepdims=True)
    return T_ji


def calculate_expected_likelihood(x_i, T_ji, mu_j, sigma_j, tau_j):
    val = 0
    for i in range(T_ji.shape[1]):
        for j in range(T_ji.shape[0]):
            val += T_ji[j, i] * (np.log(tau_j[j]) - np.log(sigma_j[j]) - (x_i[i] - mu_j[j])** 2 / (2 * sigma_j[j] ** 2))

    return val


def em_algorithm(x_i, mu_j=np.array([1, 0.1]), sigma_j=np.array([0.1, 0.1]), tau_j=np.array([0.3, 0.7]), max_iterations=1e3, tol=1e-2):

    expected_likelihood = []
    for iteration in range(int(max_iterations)):
        T_ji = maximization_step(x_i, mu_j, sigma_j, tau_j)
    
        mu_j, sigma_j, tau_j = expectation_step(x_i, T_ji)
        expected_likelihood.append(calculate_expected_likelihood(x_i, T_ji, mu_j, sigma_j, tau_j))

        if iteration > 0 and np.linalg.norm(expected_likelihood[-1] - expected_likelihood[-2]) < tol:
            break

    res = {
        "mu" : mu_j,
        "sigma" : sigma_j,
        "tau" : tau_j,
        "mixture_probability" : T_ji,
        "expected_likelihood": expected_likelihood
    }

    return res
