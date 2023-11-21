from detector import Detector
from features import FeatureExtractor, Hungarian_Matching, Frame
import utils
import os

FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000003_00000_s'

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, source_folder):
        self.detector = Detector()
        self.source_folder = source_folder
        self.img_names = utils.get_filenames_from(self.source_folder, 'jpg')
    
    def track(self):
        lf: Frame = None
        for name in self.img_names[:2]:
            img_path = self.get_img_path(name)
            img = self.get_img_tensor(name)
            detection_labels = self.detector.detect_and_read_labels(img_path)
            cf = Frame(img, detection_labels[name.split('.')[0]])
            cf.masks = FeatureExtractor.crop_masks(cf)
            if lf == None:
                lf = cf
                continue
            matching = Hungarian_Matching.match(lf.masks,cf.masks)
            print()


    def get_img_tensor(self, name):
        return utils.get_img_from_file(self.get_img_path(name))
    
    def get_img_path(self, name):
        return os.path.join(self.source_folder, name)


t = Tracker(FRAMES_FOLDER)
t.track()