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
        self.name = None
        if id > 0: self.name = utils.get_filename_from_number(id)
        self.img = img
        self.img_bb = None
        if type(self.img) is np.ndarray:
            self.img_bb = BoundingBox(int(len(img[0])/2),int(len(img)/2),len(img[0]), len(img))
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
    
    def save_frame_and_bboxes_with_id(self, output_folder: str, filename: str, show_conf:bool = False, annotations_filename = "annotations.txt", annotate: bool = True):
        if annotate:
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
    
# f = Frame(1)
# f.bboxes = [BoundingBox(35+134/2,466+181/2,134,181, conf=0.8), BoundingBox(35+133/2,468+184/2,133,184, conf=0.5),
#             BoundingBox(100+20/2,100+30/2,20,30, conf=0.4),BoundingBox(120+20/2,130+30/2,20,30, conf=0.42)]
# f.apply_confluence()

