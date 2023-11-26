from detector import Detector
from features import Hungarian_Matching, Frame
import src.utils as utils
import os
from src.frame import Frame

FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000003_00000_s'

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, source_folder):
        self.detector = Detector()
        self.source_folder = source_folder
        self.img_names = utils.get_filenames_from(self.source_folder, 'jpg')
        self.matcher = Hungarian_Matching()
        

    def track(self, delete_imgs = True):
        lf: Frame = None
        
        self.create_output_folder()
        #i = 1
        #tt = 0
        for name in self.img_names:
            img_path = self.get_img_path(name)
            img = self.get_img_tensor(name)
            detection_labels = self.detector.detect_and_read_labels(img_path, conf=0.15)
            if len(detection_labels) == 0: continue
            cf = Frame(img, detection_labels[name.split('.')[0]])
            cf.apply_parallel_non_max_suppression()
            #print(f"Mean suppression time ({i}): {tt/i}s")
            #i+=1
            cf.crop_masks()
            #cf.show_masks()
            
            if lf == None:
                for i in range(len(cf.bboxes)):
                    cf.bboxes[i].id = i
                cf.save_frame_and_bboxes_with_id(os.path.join(self.output_folder, name))
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

            cf.save_frame_and_bboxes_with_id(os.path.join(self.output_folder, name))
            lf = cf
        utils.turn_imgs_into_video(self.output_folder, self.output_folder.split('/')[-1], delete_imgs=delete_imgs, fps=10)

    def create_output_folder(self):
        folder = self.source_folder.split("/")[-1]+"_0"
        self.output_folder = os.path.join("data/track", folder)
        dig_len = 1
        i = 1
        while os.path.isdir(self.output_folder):
            self.output_folder = self.output_folder[:-dig_len] + str(i)
            i+=1
            if 10 ** dig_len <= i: dig_len+=1
        os.makedirs(self.output_folder, exist_ok = True)

    def get_img_tensor(self, name):
        return utils.get_img_from_file(self.get_img_path(name))
    
    def get_img_path(self, name):
        return os.path.join(self.source_folder, name)

# FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000003_00000_s'
# t = Tracker(FRAMES_FOLDER)
# t.track()
# exit(0)