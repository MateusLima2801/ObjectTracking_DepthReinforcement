from src.frame import Frame
from src.matchers.matcher import Matcher
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils

class Feature_Matcher(Matcher):
    matcher_type: str = "Visual Features"
    
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
    
    def generate_cost_matrix(self, features1: Frame, features2: Frame, normalize: bool = False):
        kp1, des1 = self.detect_keypoints(features1.masks)
        kp2, des2 = self.detect_keypoints(features2.masks)
        profit = np.zeros((len(features1.masks), len(features2.masks)))
        for i in range(len(profit)):
            for j in range(len(profit[0])):
                matches = self.get_matches(des1[i], des2[j])
                profit[i,j] = len(matches)
                # print(f'{i}, {j}: {profit[i,j]}')
                # self.show_matches(features1.masks[i], kp1[i], features2.masks[j], kp2[j], matches, f'{i}_{j}')
        cost = -profit
        if normalize: return utils.normalize_array(cost)
        else: return cost