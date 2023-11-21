import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class Frame():
    masks: list

    def __init__(self, img, labels):
        self.img = img
        self.labels = labels
    
    def show_masks(self):
        for mask in self.masks:
            plt.subplot(1,2,1)
            plt.imshow(self.img)
            plt.subplot(1,2,2)
            plt.imshow(mask)
            plt.show(block=True)

class Hungarian_Matching():
    
    def match(features1, features2):
        x, y = features1, features2
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        scores = np.einsum("nc,mc->nm", x, y)

        n_x, _ = scores.shape
        matching = -1 * np.ones(n_x, dtype=np.int32)

        # hungarian method
        row_ind, col_ind = opt.linear_sum_assignment(-scores)

        for (i, j) in zip(row_ind, col_ind):
            if scores[i, j] > 0* 1.4 * np.median(scores):
                matching[i] = j

        return matching

class FeatureExtractor():
    def crop_masks(frame: Frame) ->list:
        # print(img)
        # print(labels)
        masks = []
        for label in frame.labels:
            H, W, _ = frame.img.shape
            object = [int(label[1]*W), int(label[2]*H), int(label[3]*W), int(label[4]*H)]
            x,y,w,h = object
            mask = frame.img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            # self._show_mask(img, mask)
            masks.append(mask)
        return masks
    
# mat = Feature_Matcher()
# matrix = [[10,15,9],
#           [9,18,5],
#           [6,14,3]]
f1 = [[[1,10,2]],[[30,12,7]]]
f2 = [[[31,10,8]], [[2,9,1]]]
r = Hungarian_Matching.match(f1, f2)
#cost = mat.hungarian_matching(matrix, 'max')
#print(cost, matrix)
