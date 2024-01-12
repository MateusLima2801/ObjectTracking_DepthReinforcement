from __future__ import annotations
from src.matchers.matcher import Matcher
from src.matchers.position_matcher import Position_Matcher
from src.hungarian_matching import Hungarian_Matching
from src.frame import Frame
from src.utils import  *
from src.bounding_box import BoundingBox
from src.suppression import *

class CLEAR_Metrics:
    def __init__(self, motp:float, mota:float, misses: int, mismatches: int, false_positives: int, nr_gt_objects: int):
        self.mot_precision = motp
        self.mot_accuracy = mota
        self.misses = misses
        self.mismatches = mismatches
        self.false_positives = false_positives
        self.nr_gt_objects = nr_gt_objects
    
    def to_string(self) -> str:
         return f'TRACKING METRICS:\n\nMOTP: {self.mot_precision}\nMOTA: {self.mot_accuracy}\n\nMisses: {self.misses} ({safely_divide(self.misses,self.nr_gt_objects)})\nMismatches: {self.mismatches} ({safely_divide(self.mismatches,self.nr_gt_objects)})\nFalse Positives: {self.false_positives} ({safely_divide(self.false_positives,self.nr_gt_objects)})\nGroundTruth Objects: {self.nr_gt_objects}'

class MOT_Evaluator():
    # arg: distance threshold (in pixels)
    # returns MOTP and MOTPA
    @staticmethod
    def calculate_CLEAR_metrics(prediction: TrackingResult, groundTruth: TrackingResult, distance_threshold: float = 100, max_idx = None) -> CLEAR_Metrics:
        maximum = int(max(max(prediction.frames.keys()), max(groundTruth.frames.keys())))
        if max_idx != None:
            maximum = min(maximum, max_idx)
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
                if len(partial_dict.items()) > 0:
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
        
        mot_precision = safely_divide(matches_distances, number_of_matches)
        mot_accuracy = 1 - safely_divide(mismatches + false_positives + misses, number_of_gt_objects)
        
        return CLEAR_Metrics(mot_precision, mot_accuracy, misses, mismatches, false_positives, number_of_gt_objects)
    
    @staticmethod
    def get_id_mapping(ground_truth_bbs: list[BoundingBox], prediction_bbs: list[BoundingBox], threshold: float):
        mapping: dict = {}
        if len(ground_truth_bbs) > 0  and len(prediction_bbs) > 0:
            cost = Position_Matcher.generate_cost_matrix_bb(ground_truth_bbs, prediction_bbs, normalize=False)
            matching = Hungarian_Matching.match_from_cost_matrix(cost)
            
            for i,m in enumerate(matching):
                if m != -1 and  cost[i,m] <= threshold:
                    mapping[ground_truth_bbs[i].id] = prediction_bbs[m].id
        return mapping
    
    @staticmethod
    def save_results_to_file(output_path, metrics: CLEAR_Metrics, weights: list[float], conf: float, suppression: Suppression, std: list[float], matchers: list[Matcher], max_idx: int):
        f = open(output_path, "w")
        f.write(metrics.to_string())
        weights_str = "\n\nWeights:\n"
        for i, mat in enumerate(matchers): weights_str += f'\n{mat.matcher_type}: {weights[i]}'
        f.write(weights_str)
        f.write(f"\n\nConf: {conf}")
        f.write(f"\n\nSuppression: {suppression.suppression_type} (Mean time: {suppression.mean_supp_time}s)")
        std_str = "\n\nStandard Deviation Weights:\n"
        for i, mat in enumerate(matchers): std_str += f'\n{mat.matcher_type}: {std[i]}'
        f.write(std_str)
        f.write(f"\n\nMax idx: {max_idx}")
        f.close()
        
    @staticmethod
    def evaluate_annotations_result(prediction_filepath, ground_truth_filepath, max_idx: int) -> CLEAR_Metrics:
        prediction = TrackingResult(prediction_filepath)
        ground_truth = TrackingResult(ground_truth_filepath)

        metrics = MOT_Evaluator.calculate_CLEAR_metrics(prediction, ground_truth, max_idx=max_idx)
        print(metrics.to_string())
        return metrics
    
class TrackingResult():
    def __init__(self, annotations_file_path: str, imgs_file_path: str = ''):
        self.frames: dict[int,Frame] = {}

        f = open(annotations_file_path, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            info = line.replace('\n', '').split(',')
            info[:-1] = cast_list(info[:-1], int)
            info[-1] = float(info[-1])
            bb = BoundingBox(info[2]+int(info[4]/2), info[3]+int(info[5]/2), info[4], info[5], info[6], id=info[1])
            f_id = info[0]
            if f_id not in self.frames.keys():
                img = None
                if len(imgs_file_path) > 0 and os.path.isdir(imgs_file_path):
                    img_path = os.path.join(imgs_file_path, get_filename_from_number(f_id))
                    img = get_img_from_file(img_path)
                self.frames[f_id] = Frame(f_id, img)
            self.frames[f_id].bboxes.append(bb)

    def print_visual_result(self, output_folder: str, delete_imgs: bool = True, fps: int = 10, max_idx: int = None, gen_video: bool = True):
        imgs_folder = os.path.join(output_folder, "imgs")
        os.makedirs(imgs_folder, exist_ok=True)

        maximum = len(self.frames.values())
        if max_idx is not None:
            maximum = min(maximum, max_idx)

        for frame in self.frames.values():
            if frame.id > maximum: continue
            frame.save_frame_and_bboxes_with_id(output_folder, frame.name, annotate = False)
        if gen_video:
            turn_imgs_into_video(os.path.join(output_folder, "imgs"), output_folder.split(file_separator())[-1] + "_0", delete_imgs=delete_imgs, fps=fps)
# metrics = MOT_Evaluator.evaluate_annotations_result('data\\track\\uav0000297_02761_v_9\\annotations.txt','data\\VisDrone2019-MOT-test-dev\\annotations\\uav0000297_02761_v.txt',50)
# MOT_Evaluator.save_results_to_file(os.path.join('data\\track\\uav0000297_02761_v_9', "results.txt"), metrics, [1,1,0,1], 0.35, True, std = [1,1,1,1])