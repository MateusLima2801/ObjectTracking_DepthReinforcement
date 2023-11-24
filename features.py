import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import cv2 as cv
import src.utils as utils
import math
from src.frame import Frame

class Feature_Matcher():
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
        plt.imshow(img3)
        plt.legend([label])
        plt.show(block=True)
    
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
                    if m.distance < 0.75*n.distance:
                        good.append([m])
        return good
    
    def generate_features_cost_matrix(self, features1, features2):
        kp1, des1 = self.detect_keypoints(features1)
        kp2, des2 = self.detect_keypoints(features2)
        profit = np.zeros((len(features1), len(features2)))
        for i in range(len(profit)):
            for j in range(len(profit[0])):
                matches = self.get_matches(des1[i], des2[j])
                profit[i,j] = len(matches)
                #print(f'{i}, {j}: {cost[i,j]}')
                #self.show_matches(features1[i], kp1[i], features2[j], kp2[j], matches, f'{i}, {j}: {cost[i,j]}')
        cost = utils.normalize_array(-profit)
        return cost
    
class Position_Matcher():
    # square of centroids distance normalized
    # can be optimized to avoid calculus repetition
    def generate_distance_cost_matrix(self, f1: Frame, f2: Frame):
        cost = np.zeros((len(f1.bboxes), len(f2.bboxes)))
        rows = len(cost)
        cols = len(cost[0])
        for i in range(rows):
            for j in range(cols):
                cost[i,j] = self.calculate_distance(f1.img.shape,f2.img.shape, (f1.bboxes[i].x,f1.bboxes[i].y ), (f2.bboxes[j].x, f2.bboxes[j].y))
        return utils.normalize_array(cost)
    
    def calculate_distance(self,shape1, shape2, p1, p2):
        d = 0
        for i in range(len(p1)):
            d += (p1[i]/shape1[i]-p2[i]/shape2[i])**2
        return math.sqrt(d)

class Hungarian_Matching():
    def __init__(self):
        self.feature_matcher = Feature_Matcher()
        self.position_matcher = Position_Matcher()

    def generate_cost_matrix(self,features1: Frame, features2: Frame):
        x = 0.2
        feature_cost = self.feature_matcher.generate_features_cost_matrix(features1.masks, features2.masks)
        position_cost = self.position_matcher.generate_distance_cost_matrix(features1, features2)
        return x * feature_cost + (1-x) * (position_cost)

    def match(self, features1: Frame, features2: Frame):
        scores = self.generate_cost_matrix(features1, features2)

        n_x, _ = scores.shape
        matching = -1 * np.ones(n_x, dtype=np.int32)

        # hungarian method
        row_ind, col_ind = opt.linear_sum_assignment(scores)

        for (i, j) in zip(row_ind, col_ind):
            matching[i] = j

        return matching
    
    

# matrix = [[10,15,9],
#           [9,18,5],
#           [6,14,3]]
# f1 = utils.get_img_from_file('data/test/img0000001.jpg')
# f2 = utils.get_img_from_file('data/test/img0000014.jpg')
# mat = Hungarian_Matching()
# r = mat.match(f1, f2)
# #cost = mat.hungarian_matching(matrix, 'max')
#print(cost, matrix)