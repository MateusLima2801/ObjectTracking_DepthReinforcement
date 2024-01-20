import numpy as np
import src.utils as utils
from src.frame import Frame
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wasserstein_distance
from src.matchers.matcher import Matcher

class Depth_Distribution_Matcher(Matcher):
    matcher_type: str = "Depth Distribution"
   
    def generate_cost_matrix(self, f1: Frame, f2: Frame, normalize: bool = False):
        cost = np.zeros((len(f1.bboxes), len(f2.bboxes)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = self.calculate_distance(f1.bboxes[i].depth_array,f2.bboxes[j].depth_array)
        
        if normalize: return utils.normalize_array(cost)
        else: return cost

    def calculate_distance(self, depth_arr1, depth_arr2):
        maximum = max(depth_arr1.shape[0], depth_arr1.shape[1], depth_arr2.shape[0], depth_arr2.shape[1])
        arr1 = Depth_Distribution_Matcher.reescale_depth_array(depth_arr1, maximum)
        arr2 = Depth_Distribution_Matcher.reescale_depth_array(depth_arr2, maximum)
        return self.do_calculate_distance(arr1, arr2)
    
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
        interpolated_values[interpolated_values<0] = 0
        return interpolated_values
    
    def do_calculate_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        raise NotImplementedError
    
class Depth_Distribution_EMD_Matcher(Depth_Distribution_Matcher):
    matcher_type: str = "Depth Distribution (EMD)"
    
    def do_calculate_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        # Flatten the 2D arrays to 1D arrays
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()

        # Compute the Earth Mover's Distance
        return wasserstein_distance(arr1_flat, arr2_flat)
    
class Depth_Distribution_KL_Matcher(Depth_Distribution_Matcher):
    matcher_type: str = "Depth Distribution (KL)"
    epsilon = 1e-10
    
    def do_calculate_distance(self, arr1: np.ndarray, arr2:np.ndarray) -> float:
        # Avoiding division by zero by adding a small epsilon
        # Calculating KL Divergence
        mat = arr1
        try:
            mat = mat * np.log((arr1 + self.epsilon) / (arr2 + self.epsilon))
            div = abs(np.sum(mat))
        except:
            for i,line in enumerate(mat):
                for j, elmt in enumerate(line):
                    if np.isnan(elmt):
                        print(f'{i}, {j}: {arr1[i,j]} {arr2[i,j]}')
        return div