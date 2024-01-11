import math
import numpy as np
from src.bounding_box import BoundingBox
from src.frame import Frame
from src.matchers.matcher import Matcher
from src import utils

class Shape_Matcher(Matcher):
    @staticmethod
    def generate_cost_matrix(f1: Frame, f2: Frame, normalize: bool = False):
        cost = np.zeros((len(f1.bboxes), len(f2.bboxes)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = Shape_Matcher.calculate_distance(f1.bboxes[i],f2.bboxes[j])
        
        if normalize: return utils.normalize_array(cost)
        else: return cost
    
    @staticmethod
    def calculate_distance(bb1: BoundingBox, bb2: BoundingBox):
        return  math.sqrt( (bb1.w - bb2.w)**2 + (bb1.h - bb2.h)**2 )