from __future__ import annotations
import numpy as np

class BoundingBox():
    
    def __init__(self, x: int, y:int, w:int, h:int, conf:float=0, depth: float = 0, id: int = -1):
        self.x = x #x_centroid
        self.y = y #y_centroid
        self.w = w #width
        self.h = h #height
        self.x_ll = int(self.x - self.w/2)
        self.y_ll = int(self.y - self.h/2)
        self.x_ur = int(self.x + self.w/2)
        self.y_ur = int(self.y + self.h/2)
        self.conf = conf    
        self.id = id
        self.depth = depth

    @staticmethod
    def get_area_esc(bb: BoundingBox):
        return bb.w * bb.h
    
    @staticmethod
    def get_intersection_over_union_esc(bb1: BoundingBox, bb2: BoundingBox) -> float:
        area1 = BoundingBox.get_area_esc(bb1)
        area2 = BoundingBox.get_area_esc(bb2)

        xx = max( bb1.x_ll, bb2.x_ll )
        yy = max( bb1.y_ll, bb2.y_ll )
        aa = min( bb1.x_ur, bb2.x_ur )
        bb = min( bb1.y_ur, bb2.y_ur )
        w = max(0, aa - xx)
        h = max(0, bb-yy)

        intersection_area = w*h
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area
    
    @staticmethod
    def get_area_arr(bb: np.ndarray):
        return bb[:,:,2] * bb[:,:,3]
    
    @staticmethod
    def get_intersection_over_union_arr(bb1: np.ndarray, bb2: np.ndarray) -> float:
        area1 = BoundingBox.get_area_arr(bb1)
        area2 = BoundingBox.get_area_arr(bb2)

        xx = np.maximum( bb1[:,:,0]- bb1[:,:,2]/2, bb2[:,:,0]- bb2[:,:,2]/2 )
        yy = np.maximum( bb1[:,:,1]- bb1[:,:,3]/2, bb2[:,:,1]- bb2[:,:,3]/2 )
        aa = np.minimum( bb1[:,:,0]+ bb1[:,:,2]/2, bb2[:,:,0]+ bb2[:,:,2]/2 )
        bb = np.minimum( bb1[:,:,1]+ bb1[:,:,3]/2, bb2[:,:,1]+ bb2[:,:,3]/2 )
        w = np.maximum(0, aa - xx)
        h = np.maximum(0, bb - yy)

        intersection_area = w*h
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area
    
    