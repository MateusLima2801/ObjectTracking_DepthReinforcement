import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import cv2 as cv
import utils

class Frame():
    masks: list

    def __init__(self, img, labels):
        self.img = img
        self.labels = labels
    
    def crop_masks(self):
        # print(img)
        # print(labels)
        masks = []
        for label in self.labels:
            x,y,w,h = utils.get_bbox_dimensions(self.img, label)
            mask = self.img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            # self._show_mask(img, mask)
            masks.append(mask)
        self.masks = masks
    

    
    def show_masks(self):
        for mask in self.masks:
            plt.subplot(1,2,1)
            plt.imshow(self.img)
            plt.subplot(1,2,2)
            plt.imshow(mask)
            plt.show(block=True)

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
    def get_matches_amount(self, des1, des2):
        matches = self.bf.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return good
    
    def generate_features_cost_matrix(self, features1, features2):
        kp1, des1 = self.detect_keypoints(features1)
        kp2, des2 = self.detect_keypoints(features2)
        cost = np.zeros((len(features1), len(features2)))
        for i in range(len(cost)):
            for j in range(len(cost[0])):
                matches = self.get_matches_amount(des1[i], des2[j])
                cost[i,j] = len(matches)
                #print(f'{i}, {j}: {cost[i,j]}')
                #self.show_matches(features1[i], kp1[i], features2[j], kp2[j], matches, f'{i}, {j}: {cost[i,j]}')
        return cost

class Hungarian_Matching():
    def __init__(self):
        self.matcher = Feature_Matcher()

    def generate_cost_matrix(self,features1, features2):
        return self.matcher.generate_features_cost_matrix(features1, features2)

    def match(self, features1, features2):
        scores = self.generate_cost_matrix(features1, features2)

        n_x, _ = scores.shape
        matching = -1 * np.ones(n_x, dtype=np.int32)

        # hungarian method
        row_ind, col_ind = opt.linear_sum_assignment(-scores)

        for (i, j) in zip(row_ind, col_ind):
            if scores[i, j] > 0* 1.4 * np.median(scores):
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
