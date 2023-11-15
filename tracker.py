from detector import Detector
from utils import *

FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000003_00000_s'

# at first let's do detections each iteration if it works we can do detections offline before iterations
class Tracker:
    def __init__(self, source_folder):
        self.detector = Detector()
        self.source_folder = source_folder
        self.imgs = get_filenames_from(self.source_folder, 'jpg')
    
    def track(self):
        # for fr in self.imgs:
        img = self.imgs[0]
        img_path = os.path.join(self.source_folder, img)
        detection_labels = self.detector.detect_and_read_labels(img_path)
        
