from __future__ import annotations
from src.frame import SimpleFrame
from src.utils import  *
from src.bounding_box import BoundingBox

class MOTEvaluator():
    # returns MOTA and MOTP
    @staticmethod
    def calculate_accuracy(t1: TrackingResult, t2: TrackingResult) -> float | float:
        pass

class TrackingResult():
    def __init__(self, annotations_file_path: str):
        self.frames: dict[int,SimpleFrame]

        f = open(annotations_file_path, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            info = cast_list(line.split(','), int)
            bb = BoundingBox(int((info[2]+info[4])/2), int((info[3]+info[5])/2), info[4], info[5], info[6])
            if info[0] not in self.frames.keys():
                self.frames[info[0]] = SimpleFrame()
            self.frames[info[0]].bboxes.append(bb)


            