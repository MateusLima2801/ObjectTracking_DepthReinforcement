import numpy as np
from src.frame import Frame
from src.matchers.matcher import Matcher
from src import utils

class Depth_Matcher(Matcher):
    @staticmethod
    def generate_cost_matrix(f1: Frame, f2: Frame, normalize: bool = False):
        cost = np.zeros((len(f1.bboxes), len(f2.bboxes)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = Depth_Matcher.calculate_distance(f1.bboxes[i].depth,f2.bboxes[j].depth)
        
        if normalize: return utils.normalize_array(cost)
        else: return cost
    
    @staticmethod
    def calculate_distance(depth1, depth2):
        return  abs(depth1 - depth2)