from time import time
import numpy as np
from bounding_box import BoundingBox
from frame import Frame

class Suppression():
    suppression_type: str

    @staticmethod
    def apply_suppression(frame: Frame) -> float:
        start = time()
        Suppression.do_apply_suppression()
        end = time()
        return start - end
    
    @staticmethod
    def do_apply_suppression(frame: Frame):
        raise NotImplementedError
    
class EmptySuppression(Suppression):
    suppression_type = "None"

    @staticmethod
    def do_apply_suppression(frame: Frame):
        pass
    
class NMS(Suppression):
    suppression_type = "Non-Maximum Suppression"
    # O(n^2)
    @staticmethod
    def do_apply_suppression(frame: Frame):
        if len(frame.bboxes) <= 1: return 
        keep: list[BoundingBox] = []
        frame.bboxes = sorted(frame.bboxes, key=lambda bb: bb.conf, reverse=True)
        rem = []
        while len(frame.bboxes) > 0:
            keep.append(frame.bboxes.pop(0))
            for i in range(len(frame.bboxes)):
                iou = BoundingBox.get_intersection_over_union_esc(keep[-1],frame.bboxes[i])
                if iou > Frame.THRESHOLD_IOU:
                    rem.append(i)
            rem.reverse()
            for i in rem:
                frame.bboxes.pop(i)
            rem.clear()
        frame.bboxes = keep

class ParallelNMS(Suppression):
    suppression_type = "Parallel Non-Maximum Suppression"
    # O(n)
    def do_apply_suppression(frame: Frame):
        if len(frame.bboxes) <= 1: return
        n = len(frame.bboxes)
        s = list(map(lambda bb: bb.conf, frame.bboxes))
        b = list(map(lambda bb: [bb.x, bb.y, bb.w, bb.h], frame.bboxes))
        
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
                frame.bboxes.pop(i)

class Confluence(Suppression):
    suppression_type = "Confluence"

    @staticmethod
    def apply_suppression(frame: Frame):
        if len(frame.bboxes) <= 1: return
        CONFLUENCE_THRESHOLD = 1
        bbs_proximity = {}
        bbs_neighbours = {}
        new_bboxes = []
        old_bboxes = {i:bb for i,bb in enumerate(frame.bboxes)}

        for i, bb in enumerate(frame.bboxes):
            prox_sum = 0
            bbs_neighbours[i] = []
            for i_other, bb_other in enumerate(frame.bboxes):
                if bb_other == bb: continue
                prox = Confluence.calculate_normalized_confluence(frame, bb, bb_other)
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
                if neightbour_id in old_bboxes:
                    old_bboxes.pop(neightbour_id)
                    bbs_proximity.pop(neightbour_id)

        frame.bboxes = new_bboxes

    @staticmethod
    def calculate_normalized_confluence(frame: Frame, bb1: BoundingBox, bb2: BoundingBox):
        x_set = [bb1.x_ur, bb1.x_ll, bb2.x_ur, bb2.x_ll]
        y_set = [bb1.y_ur, bb1.y_ll, bb2.y_ur, bb2.y_ll]
        x_ur_1, y_ur_1 = Confluence.normalize_confluence_pair(frame, bb1.x_ur, bb1.y_ur, x_set, y_set)
        x_ll_1, y_ll_1 = Confluence.normalize_confluence_pair(frame, bb1.x_ll, bb1.y_ll, x_set, y_set)
        x_ur_2, y_ur_2 = Confluence.normalize_confluence_pair(frame, bb2.x_ur, bb2.y_ur, x_set, y_set)
        x_ll_2, y_ll_2 = Confluence.normalize_confluence_pair(frame, bb2.x_ll, bb2.y_ll, x_set, y_set)
        return abs(x_ll_2 - x_ll_1)+abs(x_ur_2 -x_ur_1)+abs(y_ur_2-y_ur_1)+abs(y_ll_2 - y_ll_1)
    
    @staticmethod
    def normalize_confluence_pair(frame: Frame, x:float, y:float, x_set:list[float], y_set:list[float]) -> float | float:
        return (x-min(x_set))/(max(x_set)-min(x_set)) ,(y-min(y_set))/(max(y_set) - min(y_set)) 
