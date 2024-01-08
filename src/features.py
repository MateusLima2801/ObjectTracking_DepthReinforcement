import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import cv2 as cv
from src.bounding_box import BoundingBox
import src.utils as utils
import math
from src.frame import Frame

class Matcher():
    def generate_cost_matrix(self, f1: Frame, f2: Frame, normalize: bool = False):
        pass

class Feature_Matcher(Matcher):
    def __init__(self):
        self.sift = cv.SIFT_create()
        self.bf = cv.BFMatcher()

    def detect_keypoints(self,imgs):
        kp,des = [],[]
        for img in imgs:
            kp1, des1 = self.sift.detectAndCompute(img,None)
            kp.append(kp1)
            des.append(des1)
        return kp,des
    
    def show_matches(self,img1,kp1,img2,kp2,matches, label):
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matches_path  = os.path.join('data', 'matches')
        os.makedirs(matches_path, exist_ok=True)
        if len(matches) > 2:
            plt.imsave(os.path.join(matches_path,''f'match{label}.jpg'), img3)
        plt.imshow(img3)
        plt.legend([label])
        # plt.show(block=True)
    
    # use Lowe ratio test to ensure that the best match is distinguished enough of the second match
    def get_matches(self, des1: np.ndarray, des2: np.ndarray):
        good = []

        # case there aren't any descriptors in an image
        if type(des1) == np.ndarray and type(des2) == np.ndarray: 
            matches = self.bf.knnMatch(des1,des2,k=2)
            for match in matches:
                # case that there aren't second best matches
                if len(match) == 1: 
                    good.append([match[0]])
                else:
                    m,n = match
                    if m.distance < 0.75*n.distance or len(good)<20:
                        good.append([m])
        return good
    
    def generate_cost_matrix(self, features1: Frame, features2: Frame, normalize: bool = False):
        kp1, des1 = self.detect_keypoints(features1.masks)
        kp2, des2 = self.detect_keypoints(features2.masks)
        profit = np.zeros((len(features1.masks), len(features2.masks)))
        for i in range(len(profit)):
            for j in range(len(profit[0])):
                matches = self.get_matches(des1[i], des2[j])
                profit[i,j] = len(matches)
                print(f'{i}, {j}: {profit[i,j]}')
                self.show_matches(features1.masks[i], kp1[i], features2.masks[j], kp2[j], matches, f'{i}_{j}')
        cost = -profit
        if normalize: return utils.normalize_array(cost)
        else: return cost
    
class Position_Matcher(Matcher):
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

class Hungarian_Matching():
    def __init__(self):
        self.feature_matcher = Feature_Matcher()
        self.matchers: list[Matcher] = [self.feature_matcher, Position_Matcher, Depth_Matcher, Shape_Matcher]

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
