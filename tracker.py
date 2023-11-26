from detector import Detector
from features import Hungarian_Matching, Frame
from src.frame import Frame
from src.midas_loader import Midas
import src.utils as utils
import os

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, matcher: Hungarian_Matching, midas: Midas):
        self.matcher = matcher
        self.midas = midas
        

    def track(self, source_folder:str, delete_imgs:bool = True, fps: float = 10.0, max_idx: int = None):
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
            detection_labels = Detector.detect_and_read_labels(img_path, conf=0.15)
            if len(detection_labels) == 0: continue
            # self.midas
            cf = Frame(img, detection_labels[name.split('.')[0]])
            cf.apply_parallel_non_max_suppression()
            #print(f"Mean suppression time ({i}): {tt/i}s")
            #i+=1
            cf.crop_masks()
            #cf.show_masks()
            
            if lf == None:
                for i in range(len(cf.bboxes)):
                    cf.bboxes[i].id = i
                cf.save_frame_and_bboxes_with_id(os.path.join(output_folder, name))
                lf = cf
                continue

            matching = self.matcher.match(lf,cf)
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

            cf.save_frame_and_bboxes_with_id(os.path.join(output_folder, name))
            lf = cf
        utils.turn_imgs_into_video(output_folder, output_folder.split('/')[-1], delete_imgs=delete_imgs, fps=fps)

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
        os.makedirs(output_folder, exist_ok = True)
        return output_folder

    @staticmethod
    def get_img_tensor(source_folder:str, name:str):
        return utils.get_img_from_file(os.path.join(source_folder, name))

# FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000003_00000_s'
# t = Tracker(FRAMES_FOLDER)
# t.track()
# exit(0)