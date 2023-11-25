from __future__ import annotations

class BoundingBox():
    def __init__(self, x: float, y:float, w:float, h:float, conf:float):
        self.x = x #x_centroid
        self.y = y #y_centroid
        self.w = w #width
        self.h = h #height
        self.x_ll = int(self.x - self.w/2)
        self.y_ll = int(self.y - self.h/2)
        self.x_ur = int(self.x + self.w/2)
        self.y_ur = int(self.y + self.h/2)
        self.conf = conf    
        self.id = -1

    def get_area(self):
        return self.w * self.h
     
    def get_intersection_over_union(self, other: BoundingBox) -> float:
        area1 = self.get_area()
        area2 = other.get_area()

        xx = max( self.x_ll, other.x_ll )
        yy = max( self.y_ll, other.y_ll )
        aa = min( self.x_ur, other.x_ur )
        bb = min( self.y_ur, other.y_ur )
        w = max(0, aa - xx)
        h = max(0, bb-yy)

        intersection_area = w*h
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area