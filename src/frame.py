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

    def __init__(self,id:int, img: np.ndarray = None, read_labels: list[list[float]] = None, depth_array: np.ndarray = None):
        self.id = id
        self.name = None
        if id > 0: self.name = utils.get_filename_from_number(id)
        self.img = img
        self.depth_array = depth_array
        if read_labels != None:
            self.bboxes = self.init_bboxes(read_labels, self.depth_array)
        else: self.bboxes: list[BoundingBox] = []
        self.masks: list

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
    
    def init_bboxes(self, read_labels: list[list[float]], depth_array: np.ndarray) -> list[BoundingBox]:
        bboxes = []
        for label in read_labels:
            bboxes.append(BoundingBox(label[0],label[1],label[2],label[3], label[4],depth_array=depth_array))
        return bboxes
    
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

