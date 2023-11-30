from MOT_evaluator import CLEAR_Metrics, MOT_Evaluator, TrackingResult
from detector import Detector
from features import Hungarian_Matching, Frame
from src.frame import Frame
from src.midas_loader import Midas
import src.utils as utils
import os
import numpy as np

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, matcher: Hungarian_Matching, midas: Midas, detector: Detector):
        self.matcher = matcher
        self.midas = midas
        self.detector = detector

    def track(self, source_folder:str, weights: list[float] = [1/3,1/3,1/3], delete_imgs:bool = True,  fps: float = 10.0, max_idx: int = None, ground_truth_filepath = None, conf = 0.6):
        lf: Frame = None
        img_names = utils.get_filenames_from(source_folder, 'jpg')
        output_folder = Tracker.create_output_folder(source_folder)
        #i = 1
        #tt = 0
        if max_idx is not None:
            img_names = img_names[:max_idx]
        for name in img_names:
            img_path = os.path.join(source_folder, name)
            img = Tracker.get_img_tensor(source_folder, name)
            detection_labels = self.detector.detect_and_read_labels(img_path, conf)
            if len(detection_labels) <= 0: continue
            depth_array = np.zeros((img.shape[0], img.shape[1], 1))
            if weights[2] > 0:
                depth_array = self.midas.get_depth_array(img)
            id = utils.get_number_from_filename(name)
            cf = Frame(id,img, detection_labels, depth_array)
            cf.apply_parallel_non_max_suppression()
            #print(f"Mean suppression time ({i}): {tt/i}s")
            #i+=1
            
            cf.crop_masks()
            #cf.show_masks()
            
            if lf == None:
                for i in range(len(cf.bboxes)):
                    cf.bboxes[i].id = i
                cf.save_frame_and_bboxes_with_id(output_folder, name)
                lf = cf
                continue

            matching = self.matcher.match(lf,cf,weights)
            #print(matching)
            # if matching == -1: doesn't have a match
            for i in range(len(matching)):
                if matching[i] >=0 :
                    cf.bboxes[matching[i]].id = lf.bboxes[i].id
            free_id = max(list(map(lambda x: x.id, cf.bboxes))) + 1
            for i in range(len(cf.bboxes)):
                if cf.bboxes[i].id == -1:
                    cf.bboxes[i].id = free_id
                    free_id +=1

            cf.save_frame_and_bboxes_with_id(output_folder, name)
            lf = cf
        utils.turn_imgs_into_video(os.path.join(output_folder, "imgs"), output_folder.split('/')[-1], delete_imgs=delete_imgs, fps=fps)
        
        if ground_truth_filepath != None:
            metrics = MOT_Evaluator.evaluate_annotations_result(os.path.join(output_folder,'annotations.txt'), ground_truth_filepath, max_idx)
            MOT_Evaluator.save_results_to_file(os.path.join(output_folder, "results.txt"), metrics, weights, conf)
    
    @staticmethod
    def create_output_folder(source_folder: str)  -> str:
        folder = source_folder.split("/")[-1]+"_0"
        output_folder = os.path.join("data/track", folder)
        dig_len = 1
        i = 1
        while os.path.isdir(output_folder):
            output_folder = output_folder[:-dig_len] + str(i)
            i+=1
            if 10 ** dig_len <= i: dig_len+=1
        os.makedirs(os.path.join(output_folder, "imgs"), exist_ok = True)
        return output_folder

    @staticmethod
    def get_img_tensor(source_folder:str, name:str):
        return utils.get_img_from_file(os.path.join(source_folder, name))
# FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000003_00000_s'
# t = Tracker(FRAMES_FOLDER)
# t.track()
# exit(0)