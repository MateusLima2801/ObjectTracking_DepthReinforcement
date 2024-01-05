from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from time import time 
import sys
sys.path.append('/home/mateus/Desktop/ObjectTracking_DepthReinforcements')
import src.utils as utils
from src.bounding_box import BoundingBox

class Frame():
    THRESHOLD_IOU = 0.7
    MAX_AGE = 10

    def __init__(self,id:int, img: np.ndarray = None, read_labels: list[list[float]] = None, depth_array: np.ndarray = None):
        self.id = id
        self.img = img
        self.depth_array = depth_array
        if read_labels != None:
            self.bboxes = self.init_bboxes(read_labels)
        else: self.bboxes: list[BoundingBox] = []
        self.masks: list
        
        #self.apply_parallel_non_max_suppression()

    def crop_masks(self):
        # print(img)
        # print(labels)
        masks = []
        for bb in self.bboxes:
            w0, w1 = max(0,bb.x_ll),min(self.img.shape[1], bb.x_ur)
            h0, h1 = max(0,bb.y_ur), min(self.img.shape[0], bb.y_ll)
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
    
    def init_bboxes(self, read_labels: list[list[float]]) -> list[BoundingBox]:
        bboxes = []
        for label in read_labels:
            depth = float(self.depth_array[Frame.interpol(label[1],len(self.depth_array)), Frame.interpol(label[0], len(self.depth_array[0]))])
            bboxes.append(BoundingBox(label[0],label[1],label[2],label[3], label[4],depth))
        return bboxes

    def choose_virtual_bboxes(self, lf: Frame, predicted_centroids: np.ndarray):
        virtual = lf.bboxes.copy()
        for i in range(len(virtual)):
            virtual[i].update_position(x=predicted_centroids[i,0], y=predicted_centroids[i,1], virtual=True)
        virtual = list(filter(lambda bb: bb.age < Frame.MAX_AGE, virtual))
        iou_mask_current = Frame.get_iou_mask_reduced(self.bboxes, virtual)
        for i, bb in enumerate(virtual):
            if not iou_mask_current[i]:
                bb.reset_id()
                self.bboxes.append(bb)

    @staticmethod
    def get_iou_mask_reduced( bboxes: list[BoundingBox], virtual: list[BoundingBox]):
        iou = np.zeros((len(bboxes), len(virtual)))
        for i in range(len(iou)):
            for j in range(len(iou[0])):
                iou[i,j] = BoundingBox.get_intersection_over_union_esc(bboxes[i], virtual[j])
        iou_mask = iou > Frame.THRESHOLD_IOU
        iou_mask_reduced = iou_mask[0]
        for i in range(1, len(iou_mask)):
            iou_mask_reduced = np.logical_or(iou_mask_reduced,iou_mask[i])
        return iou_mask_reduced
    
    @staticmethod
    def interpol(n, maxi, mini = 0):
        return min(max(mini,n),maxi-1)
    
    def save_frame_and_bboxes_with_id(self, output_folder: str, filename: str, show_conf:bool = False, annotations_filename = "annotations.txt"):
        self.annotate_frame(output_folder, annotations_filename)

        img_output = os.path.join(output_folder, "imgs", filename)
        copy = self.img.copy()
        for bb in self.bboxes:
            cv.rectangle(copy, (bb.x_ll, bb.y_ll), (bb.x_ur, bb.y_ur), color=(255,255,0), thickness=2)
            label = f'Id: {bb.id}'
            if show_conf: label += ', Conf: '+'{:.2f}'.format(bb.conf)
            cv.putText(
                copy,
                label,
                (bb.x_ll, bb.y_ll - 10),
                fontFace = cv.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (255, 255, 255),
                thickness=2
            )
        bgr_img = cv.cvtColor(copy, cv.COLOR_RGB2BGR)
        cv.imwrite(img_output, bgr_img)

    def annotate_frame(self, output_folder: str, annotations_filename: str):
        file_path = os.path.join(output_folder, annotations_filename)

        f = open(file_path, "a")
        for bb in self.bboxes:
            info  = utils.cast_list([self.id, bb.id, bb.x_ll, bb.y_ur, bb.w, bb.h, bb.conf], str)
            line = ','.join(info) +'\n'
            f.write(line)
        f.close()

    def get_bbox_by_id(self, id):
        for bb in self.bboxes:
            if id == bb.id: return bb
        return None
    
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
    
    def apply_confluence(self):
        CONFLUENCE_THRESHOLD = 1
        bbs_proximity = {}
        bbs_neighbours = {}
        new_bboxes = []
        old_bboxes = {i:bb for i,bb in enumerate(self.bboxes)}

        for i, bb in enumerate(self.bboxes):
            prox_sum = 0
            bbs_neighbours[i] = []
            for i_other, bb_other in enumerate(self.bboxes):
                if bb_other == bb: continue
                prox = self.calculate_normalized_confluence(bb, bb_other)
                if prox < CONFLUENCE_THRESHOLD:
                    prox_sum += prox
                    bbs_neighbours[i].append(i_other)
            bbs_proximity[i] = prox_sum*(1-bb.conf)
            if len(bbs_neighbours[i]) > 0:
                 bbs_proximity[i]/=len(bbs_neighbours[i])
        
        while len(old_bboxes.values()) > 0:
            bb_idx = min(bbs_proximity, key=bbs_proximity.get)
            bb = old_bboxes.pop(bb_idx)
            bbs_proximity.pop(bb_idx)
            new_bboxes.append(bb)
            for neightbour_id in bbs_neighbours[bb_idx]:
                old_bboxes.pop(neightbour_id)

        self.bboxes = new_bboxes

    def calculate_normalized_confluence(self, bb1: BoundingBox, bb2: BoundingBox):
        x_set = [bb1.x_ur, bb1.x_ll, bb2.x_ur, bb2.x_ll]
        y_set = [bb1.y_ur, bb1.y_ll, bb2.y_ur, bb2.y_ll]
        x_ur_1, y_ur_1 = self.normalize_confluence_pair(bb1.x_ur, bb1.y_ur, x_set, y_set)
        x_ll_1, y_ll_1 = self.normalize_confluence_pair(bb1.x_ll, bb1.y_ll, x_set, y_set)
        x_ur_2, y_ur_2 = self.normalize_confluence_pair(bb2.x_ur, bb2.y_ur, x_set, y_set)
        x_ll_2, y_ll_2 = self.normalize_confluence_pair(bb2.x_ll, bb2.y_ll, x_set, y_set)
        return abs(x_ll_2 - x_ll_1)+abs(x_ur_2 -x_ur_1)+abs(y_ur_2-y_ur_1)+abs(y_ll_2 - y_ll_1)
    
    def normalize_confluence_pair(self, x:float, y:float, x_set:list[float], y_set:list[float]) -> float | float:
        return (x-min(x_set))/(max(x_set)-min(x_set)) ,(y-min(y_set))/(max(y_set) - min(y_set)) 

f = Frame(1)
f.bboxes = [BoundingBox(35+134/2,466+181/2,134,181, conf=0.8), BoundingBox(35+133/2,468+184/2,133,184, conf=0.5),
            BoundingBox(100+20/2,100+30/2,20,30, conf=0.4),BoundingBox(120+20/2,130+30/2,20,30, conf=0.42)]
f.apply_confluence()

