import shutil
from src.MOT_evaluator import MOT_Evaluator
from src.detector import Detector
from src.hungarian_matching import Hungarian_Matching
from src.frame import Frame
from src.midas_loader import Midas
from progress.bar import Bar
from src.suppression import *
from tqdm import tqdm
import src.utils as utils
import os
import numpy as np

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, matcher: Hungarian_Matching, midas: Midas, detector: Detector):
        self.matcher = matcher
        self.midas = midas
        self.detector = detector

    def track(self, source_folder:str, depth_base_folder: str, weights: list[float] = [1,1,1,1,1],
              delete_imgs:bool = True,  fps: float = 10.0, max_idx: int = None, 
              ground_truth_filepath = None, conf = 0.6, suppression: Suppression = EmptySuppression(), std_deviations = [1,1,1,1,1],
              decompress= True):
        lf: Frame = None
        img_names = utils.get_filenames_from(source_folder, 'jpg')
        output_folder = Tracker.create_output_folder(source_folder)
        
        sequence_name = source_folder.split(utils.file_separator())[-1]
        if weights[2] > 0 or weights[4] > 0:
            depth_source_folder = os.path.join(depth_base_folder, sequence_name)
            os.makedirs(depth_base_folder, exist_ok=True)
            if f'{sequence_name}.tar.gz' in os.listdir(depth_base_folder) and decompress:
                utils.decompress_file(os.path.join(depth_base_folder,f'{sequence_name}.tar.gz'), depth_source_folder)
            os.makedirs(depth_source_folder, exist_ok=True)
        
        if max_idx is not None:
            img_names = img_names[:max_idx]
        bar = tqdm(total=len(img_names))
        suppression.init_time_count()
        for name in img_names:
            img_path = os.path.join(source_folder, name)
            img = utils.get_img_from_file(img_path)
            detection_labels = self.detector.detect_and_read_labels(img_path, conf)
            if len(detection_labels) <= 0: continue
            depth_array = np.zeros((img.shape[0], img.shape[1], 1))
            if weights[2] > 0 or weights[4] > 0:
                # depth_array = self.midas.get_depth_array(img)
                depth_array = self.midas.try_get_or_create_depth_array(img, name, depth_source_folder)
            id = utils.get_number_from_filename(name)
            cf = Frame(id,img, detection_labels, depth_array)
            suppression.apply_suppression(cf)
            
            if lf == None:
                cf.crop_masks()
                #cf.show_masks()
                for i in range(len(cf.bboxes)):
                    cf.bboxes[i].id = i
                cf.save_frame_and_bboxes_with_id(output_folder, name)
                lf = cf
                bar.update(1)
                bar.set_postfix({sequence_name: id})
                bar.refresh()
                continue

            cf.crop_masks()
            #cf.show_masks()
            
            matching = self.matcher.match(lf,cf,weights, std_deviations)
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
            bar.update(1)
            bar.set_postfix({sequence_name: id})
            bar.refresh()
        suppression.end_time_count()
        if weights[2] > 0 or weights[4] > 0:
            utils.compress_folder(depth_source_folder)
            shutil.rmtree(depth_source_folder)
        utils.turn_imgs_into_video(os.path.join(output_folder, "imgs"), output_folder.split(utils.file_separator())[-1], delete_imgs=delete_imgs, fps=fps)
        
        if ground_truth_filepath != None:
            metrics = MOT_Evaluator.evaluate_annotations_result(os.path.join(output_folder,'annotations.txt'), ground_truth_filepath, max_idx)
            MOT_Evaluator.save_results_to_file(os.path.join(output_folder, "results.txt"), metrics, weights, conf, suppression, std_deviations, self.matcher.matchers, max_idx)
            return metrics
        return None
    
    @staticmethod
    def create_output_folder(source_folder: str)  -> str:
        folder = source_folder.split(utils.file_separator())[-1]+"_0"
        output_folder = os.path.join("data", "track", folder)
        dig_len = 1
        i = 1
        while os.path.isdir(output_folder):
            output_folder = output_folder[:-dig_len] + str(i)
            i+=1
            if 10 ** dig_len <= i: dig_len+=1
        os.makedirs(os.path.join(output_folder, "imgs"), exist_ok = True)
        return output_folder