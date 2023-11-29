from __future__ import annotations
from features import Hungarian_Matching, Position_Matcher
from src.frame import Frame
from src.utils import  *
from src.bounding_box import BoundingBox

class CLEAR_Metrics:
    def __init__(self, motp:float, mota:float, misses: int, mismatches: int, false_positives: int, nr_gt_objects: int):
        self.mot_precision = motp
        self.mot_accuracy = mota
        self.misses = misses
        self.mismatches = mismatches
        self.false_positives = false_positives
        self.nr_gt_objects = nr_gt_objects
    
    def print(self):
         print(f'TRACKING METRICS:\n\nMOTP: {self.mot_precision}\nMOTA: {self.mot_accuracy}\n\nMisses: {self.misses}\nMismatches: {self.mismatches}\nFalse Positives: {self.false_positives}\nGroundTruth Objects: {self.nr_gt_objects}')

class MOT_Evaluator():
    # arg: distance threshold (in pixels)
    # returns MOTP and MOTPA
    @staticmethod
    def calculate_CLEAR_metrics(prediction: TrackingResult, groundTruth: TrackingResult, distance_threshold: float = 50) -> CLEAR_Metrics:
        maximum = int(max(max(prediction.frames.keys()), max(groundTruth.frames.keys())))
        prev_id_mapping: dict = {}
        mismatches = 0
        false_positives = 0
        misses = 0
        number_of_gt_objects = 0
        number_of_matches = 0
        matches_distances = 0
        for i in range(1,maximum):
            in_prediction = i in prediction.frames.keys()
            in_ground_truth = i in groundTruth.frames.keys()

            if not in_prediction and in_ground_truth:
                misses += len(groundTruth.frames[i].bboxes)
                number_of_gt_objects += len(groundTruth.frames[i].bboxes)
            elif in_prediction and not in_ground_truth:
                false_positives += len(prediction.frames[i].bboxes)
            elif in_prediction and in_ground_truth :
                id_mapping: dict = {}
                for key, val in prev_id_mapping.items():
                     bb_gt = groundTruth.frames[i].get_bbox_by_id(key)
                     bb_pred = prediction.frames[i].get_bbox_by_id(val)
                     
                     if bb_gt != None and bb_pred != None:
                        dist = Position_Matcher.calculate_distance(bb_gt, bb_pred)
                        if dist < distance_threshold:
                            id_mapping[key] = val
                
                gt_bbs = list(filter(lambda bb: bb.id not in id_mapping.keys(),groundTruth.frames[i].bboxes))
                pred_bbs = list(filter(lambda bb: bb.id not in id_mapping.values(),prediction.frames[i].bboxes)) 
                partial_dict = MOT_Evaluator.get_id_mapping(gt_bbs, pred_bbs, distance_threshold)
                
                #increment mismatches
                if len(prev_id_mapping.items()) > 0:
                    mismatches += sum(1 for key, val in partial_dict.items() if key in prev_id_mapping.keys() and val != prev_id_mapping[key])
                id_mapping.update(partial_dict)

                #increment MOTP metrics
                number_of_matches += len(id_mapping.items())
                for key, val in id_mapping.items():
                    bb_gt = groundTruth.frames[i].get_bbox_by_id(key)
                    bb_pred = prediction.frames[i].get_bbox_by_id(val)
                    matches_distances += Position_Matcher.calculate_distance(bb_gt, bb_pred)
                
                #increment false positives and misses
                false_positives += sum(1 for bb in prediction.frames[i].bboxes if bb.id not in id_mapping.values())
                misses += sum(1 for bb in groundTruth.frames[i].bboxes if bb.id not in id_mapping.keys())
 
                number_of_gt_objects += len(groundTruth.frames[i].bboxes)
                prev_id_mapping = id_mapping
        mot_precision = matches_distances / number_of_matches
        mot_accuracy = 1 - ( (mismatches + false_positives + misses) / number_of_gt_objects)
        return CLEAR_Metrics(mot_precision, mot_accuracy, misses, mismatches, false_positives, number_of_gt_objects)
    
    @staticmethod
    def get_id_mapping(ground_truth_bbs: list[BoundingBox], prediction_bbs: list[BoundingBox], threshold: float):
        mapping: dict = {}
        if len(ground_truth_bbs) > 0  and len(prediction_bbs) > 0:
            cost = Position_Matcher.generate_cost_matrix(ground_truth_bbs, prediction_bbs, normalize=False)
            matching = Hungarian_Matching.match_from_cost_matrix(cost)
            
            for i,m in enumerate(matching):
                if m != -1 and  cost[i,m] <= threshold:
                    mapping[ground_truth_bbs[i].id] = prediction_bbs[m].id
        return mapping
    
class TrackingResult():
    def __init__(self, annotations_file_path: str):
        self.frames: dict[int,Frame] = {}

        f = open(annotations_file_path, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            info = line.replace('\n', '').split(',')
            info[:-1] = cast_list(info[:-1], int)
            info[-1] = float(info[-1])
            bb = BoundingBox(int((info[2]+info[4])/2), int((info[3]+info[5])/2), info[4], info[5], info[6], id=info[1])
            if info[0] not in self.frames.keys():
                self.frames[info[0]] = Frame(info[0])
            self.frames[info[0]].bboxes.append(bb)


# prediction = TrackingResult('data/track/uav0000009_03358_v_2/annotations.txt')
# ground_truth = TrackingResult('data/VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt')

# metrics = MOT_Evaluator.calculate_CLEAR_metrics(prediction, ground_truth)
# metrics.print()