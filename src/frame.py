import matplotlib.pyplot as plt
import src.utils as utils
from src.bounding_box import BoundingBox
import numpy as np
import cv2 as cv
import os
from time import time 

class Frame():
    masks: list
    THRESHOLD_IOU = 0.8

    def __init__(self, img, read_labels: list[list[float]]):
        self.img = img
        self.bboxes = self.unnormalize_labels(read_labels)
        #self.apply_parallel_non_max_suppression()
        

    def crop_masks(self):
        # print(img)
        # print(labels)
        masks = []
        for bb in self.bboxes:
            w0, w1 = max(0,int(bb.x-bb.w/2)),min(self.img.shape[1], int(bb.x+bb.w/2))
            h0, h1 = max(0,int(bb.y-bb.h/2)), min(self.img.shape[0], int(bb.y+bb.h/2))
            mask = self.img[h0:h1, w0:w1]
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
    
    def unnormalize_labels(self, read_labels: list[str]) -> list[BoundingBox]:
        bboxes = []
        for i in range(len(read_labels)):
            x,y,w,h = utils.get_bbox_dimensions(self.img, read_labels[i])
            bboxes.append(BoundingBox(x,y,w,h, float(read_labels[i][5])))
        return bboxes

    def save_frame_and_bboxes_with_id(self, output_filename: str, show_conf:bool = False):
        for bb in self.bboxes:
            cv.rectangle(self.img, (bb.x_ll, bb.y_ll), (bb.x_ur, bb.y_ur), color=(255,255,0), thickness=2)
            label = f'Id: {bb.id}'
            if show_conf: label += ', Conf: '+'{:.2f}'.format(bb.conf)
            cv.putText(
                self.img,
                label,
                (bb.x_ll, bb.y_ll - 10),
                fontFace = cv.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (255, 255, 255),
                thickness=2
            )
        bgr_img = cv.cvtColor(self.img, cv.COLOR_RGB2BGR)
        cv.imwrite(output_filename, bgr_img)

    # O(n^2)
    def apply_non_max_suppression(self):
        if len(self.bboxes) <= 1: return 0
        start = time()
        keep: list[BoundingBox] = []
        self.bboxes = sorted(self.bboxes, key=lambda bb: bb.conf, reverse=True)
        rem = []
        while len(self.bboxes) > 0:
            keep.append(self.bboxes.pop(0))
            for i in range(len(self.bboxes)):
                iou = BoundingBox.get_intersection_over_union_esc(keep[-1],self.bboxes[i])
                if iou > Frame.THRESHOLD_IOU:
                    rem.append(i)
            rem.reverse()
            for i in rem:
                self.bboxes.pop(i)
            rem.clear()
        self.bboxes = keep
        end = time()
        return end - start
    # O(n)
    def apply_parallel_non_max_suppression(self):
        if len(self.bboxes) <= 1: return 0
        start = time()
        n = len(self.bboxes)
        s = list(map(lambda bb: bb.conf, self.bboxes))
        b = list(map(lambda bb: [bb.x, bb.y, bb.w, bb.h], self.bboxes))
        
        row_B, row_S = [], []
        for i in range(n): #O(n)
            row_B.append(b)
            row_S.append(s)
        row_B = np.array(row_B)
        row_S = np.array(row_S)
        col_B = row_B.transpose((1,0,2)) # O(1)
        col_S = row_S.transpose((1,0))
        
        iou = BoundingBox.get_intersection_over_union_arr(row_B, col_B)
        iou_mask = iou > Frame.THRESHOLD_IOU
        score_mask = col_S > row_S

        final_mask_matrix = np.logical_and(iou_mask, score_mask) #O(1)
        final_mask_reduced = final_mask_matrix[0]
        for i in range(1,n):
            final_mask_reduced = np.logical_or(final_mask_reduced,final_mask_matrix[i])
        
        for i in range(n-1,-1,-1):
            if final_mask_reduced[i]:
                self.bboxes.pop(i)
        end = time()
        return end - start