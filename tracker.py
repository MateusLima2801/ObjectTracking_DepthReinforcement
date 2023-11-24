from detector import Detector
from features import Hungarian_Matching, Frame
import src.utils as utils
import os
import cv2 as cv
from src.frame import Frame

FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000016_00000_s'

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, source_folder):
        self.detector = Detector()
        self.source_folder = source_folder
        self.img_names = utils.get_filenames_from(self.source_folder, 'jpg')
        self.matcher = Hungarian_Matching()
        

    def track(self):
        lf: Frame = None
        
        self.create_output_folder()
        for name in self.img_names:
            img_path = self.get_img_path(name)
            img = self.get_img_tensor(name)
            detection_labels = self.detector.detect_and_read_labels(img_path, conf=0.15)
            if len(detection_labels) == 0: continue
            cf = Frame(img, detection_labels[name.split('.')[0]])
            cf.crop_masks()
            #cf.show_masks()
            
            if lf == None:
                for i in range(len(cf.bboxes)):
                    cf.bboxes[i].id = i
                self.save_frame_and_bboxes_with_id(cf.bboxes, img, name)
                lf = cf
                continue

            matching = self.matcher.match(lf,cf)
            print(matching)
            # if matching == -1: doesn't have a match
            for i in range(len(matching)):
                if matching[i] >=0 :
                    cf.bboxes[matching[i]].id = lf.bboxes[i].id
            free_id = max(list(map(lambda x: x.id, cf.bboxes))) + 1
            for i in range(len(cf.bboxes)):
                if cf.bboxes[i].id == -1:
                    cf.bboxes[i].id = free_id
                    free_id +=1

            self.save_frame_and_bboxes_with_id(cf.bboxes, img, name)
            lf = cf
        utils.turn_imgs_into_video(self.output_folder, self.output_folder.split('/')[-1], delete_imgs=False)

    def create_output_folder(self):
        folder = self.source_folder.split("/")[-1]+"_0"
        self.output_folder = os.path.join("data/track", folder)
        dig_len = 1
        i = 1
        while os.path.isdir(self.output_folder):
            self.output_folder = self.output_folder[:-dig_len] + str(i)
            if 10 ** dig_len <= i: dig_len+=1
        os.makedirs(self.output_folder, exist_ok = True)

    def save_frame_and_bboxes_with_id(self, bboxes: list, img, name: str):
        for bb in bboxes:
            x0, y0 = int(bb.x-bb.w/2), int(bb.y-bb.h/2)
            x1, y1 = int(bb.x+bb.w/2), int(bb.y+bb.h/2)
            cv.rectangle(img, (x0,y0), (x1,y1), color=(255,255,0), thickness=2)
            
            cv.putText(
                img,
                f'Id: {bb.id}',
                (x0, y0 - 10),
                fontFace = cv.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (255, 255, 255),
                thickness=2
            )
        bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(self.output_folder, name), bgr_img)

    def get_img_tensor(self, name):
        return utils.get_img_from_file(self.get_img_path(name))
    
    def get_img_path(self, name):
        return os.path.join(self.source_folder, name)


t = Tracker(FRAMES_FOLDER)
t.track()