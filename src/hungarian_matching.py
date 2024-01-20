
import numpy as np
import scipy.optimize as opt
from src.matchers.depth_distribution_matcher import Depth_Distribution_Matcher, Depth_Distribution_KL_Matcher
from src.matchers.depth_matcher import Depth_Matcher
from src.matchers.feature_matcher import Feature_Matcher
from src.matchers.position_matcher import Position_Matcher
from src.matchers.shape_matcher import Shape_Matcher
from src.matchers.matcher import Matcher
from src.frame import Frame

class Hungarian_Matching():
    def __init__(self):
        self.matchers: list[Matcher] = [Feature_Matcher(), Position_Matcher, Depth_Matcher, Shape_Matcher, Depth_Distribution_KL_Matcher()]

    def generate_cost_matrix(self,fr1: Frame, fr2: Frame, weights: list[float], std_deviations: list[float]):
        cost = np.zeros((len(fr1.bboxes), len(fr2.bboxes)))
        if len(weights) > len(self.matchers):
            raise Exception("Wrong weights: "+ str(weights))
        for i, w in enumerate(weights):
            if w > 0:
                partial_cost = (w/std_deviations[i]) * self.matchers[i].generate_cost_matrix(fr1, fr2)
                cost += partial_cost
        return cost

    def match(self, features1: Frame, features2: Frame, weights: list[float], std_deviations: list[float]):
        scores = self.generate_cost_matrix(features1, features2, weights, std_deviations)

        n_x, _ = scores.shape
        matching = -1 * np.ones(n_x, dtype=np.int32)

        # hungarian method
        row_ind, col_ind = opt.linear_sum_assignment(scores)

        for (i, j) in zip(row_ind, col_ind):
            matching[i] = j

        return matching
    
    @staticmethod
    def match_from_cost_matrix(scores: np.ndarray):
        n_x, _ = scores.shape
        matching = -1 * np.ones(n_x, dtype=np.int32)

        # hungarian method
        row_ind, col_ind = opt.linear_sum_assignment(scores)

        for (i, j) in zip(row_ind, col_ind):
            matching[i] = j

        return matching
