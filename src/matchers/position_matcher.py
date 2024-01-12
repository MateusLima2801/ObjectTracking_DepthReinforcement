import math
import numpy as np
from src.bounding_box import BoundingBox
from src.frame import Frame
from src.matchers.matcher import Matcher
from src import utils

class Position_Matcher(Matcher):
    matcher_type: str = "Position"
    
    # square of centroids distance normalized
    # can be optimized to avoid calculus repetition
    @staticmethod
    def generate_cost_matrix(f1: Frame, f2: Frame, normalize: bool = False):
        cost = np.zeros((len(f1.bboxes), len(f2.bboxes)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = Position_Matcher.calculate_distance(f1.bboxes[i], f2.bboxes[j])

        if normalize: return utils.normalize_array(cost)
        else: return cost

    @staticmethod
    def generate_cost_matrix_bb(f1: list[BoundingBox], f2: list[BoundingBox], normalize: bool = True):
        cost = np.zeros((len(f1), len(f2)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = Position_Matcher.calculate_distance(f1[i], f2[j])

        if normalize: return utils.normalize_array(cost)
        else: return cost
    
    @staticmethod
    def calculate_distance(bb1: BoundingBox, bb2: BoundingBox):
        p1, p2 = (bb1.x, bb1.y), (bb2.x, bb2.y)
        d = 0
        for i in range(len(p1)):
            d += (p1[i]-p2[i])**2
        return math.sqrt(d)