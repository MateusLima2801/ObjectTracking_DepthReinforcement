import numpy as np
import src.utils as utils
from src.frame import Frame
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wasserstein_distance
from src.matchers.matcher import Matcher

class Depth_Distribution_Matcher(Matcher):
    matcher_type: str = "Depth Distribution"
    
    @staticmethod
    def generate_cost_matrix(f1: Frame, f2: Frame, normalize: bool = False):
        cost = np.zeros((len(f1.bboxes), len(f2.bboxes)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = Depth_Distribution_Matcher.calculate_distance(f1.bboxes[i].depth_array,f2.bboxes[j].depth_array)
        
        if normalize: return utils.normalize_array(cost)
        else: return cost
    
    @staticmethod
    def calculate_distance(depth_arr1, depth_arr2):
        maximum = max(depth_arr1.shape[0], depth_arr1.shape[1], depth_arr2.shape[0], depth_arr2.shape[1])
        arr1 = Depth_Distribution_Matcher.reescale_depth_array(depth_arr1, maximum)
        arr2 = Depth_Distribution_Matcher.reescale_depth_array(depth_arr2, maximum)

        # Flatten the 2D arrays to 1D arrays
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()

        # Compute the Earth Mover's Distance
        emd_distance = wasserstein_distance(arr1_flat, arr2_flat)
        return emd_distance
    
    @staticmethod
    def reescale_depth_array(arr: np.ndarray, mesh_side: int):
        x_values = np.linspace(0, arr.shape[0], arr.shape[0])
        y_values = np.linspace(0, arr.shape[1], arr.shape[1])

        # Create a bivariate spline interpolation
        interp_func = RegularGridInterpolator((x_values, y_values), arr, method='linear', bounds_error=False, fill_value=None)
        
         # Generate new points for interpolation
        new_x_values = np.linspace(0, mesh_side, 100)
        new_y_values = np.linspace(0, mesh_side, 100)
        new_points = np.array(np.meshgrid(new_x_values, new_y_values)).T.reshape(-1, 2)

        # Evaluate the interpolation function at the new points
        interpolated_values = interp_func(new_points).reshape(len(new_x_values), len(new_y_values))
        
        return interpolated_values