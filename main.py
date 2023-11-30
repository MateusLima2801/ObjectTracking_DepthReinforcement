from detector import Detector
from src.midas_loader import Midas
from tracker import Tracker
from features import Hungarian_Matching
import os

def main():
    SEQUENCES_FOLDER = 'data/VisDrone2019-SOT-train/sequences'
    SEQUENCE_FOLDER = 'data/VisDrone2019-MOT-test-dev/sequences/uav0000355_00001_v'
    GROUND_TRUTH_FILEPATH ='data/VisDrone2019-MOT-test-dev/annotations/uav0000355_00001_v.txt'
    matcher = Hungarian_Matching()
    midas = Midas()
    detector = Detector()
    tracker = Tracker(matcher, midas, detector)
    weights = [[0.3, 0.7, 0]]#, [0.5, 0, 0.5], [1/3,1/3,1/3]]
    # sequences = os.listdir(SEQUENCES_FOLDER)
    # for seq in sequences[:1]:
    #     tracker.track(seq, True, fps=10)
    
    for w in weights:
        tracker.track(SEQUENCE_FOLDER, fps=10, max_idx=10,delete_imgs=False,weights=w,ground_truth_filepath=GROUND_TRUTH_FILEPATH, conf=0.35)
    exit(0)
    #midas.transform_img('data/test/img0000001.jpg', 'data/depth/test')
    #midas.transform_imgs_from_folder('data/test', 'data/depth/test')

if __name__ == "__main__":
    main()