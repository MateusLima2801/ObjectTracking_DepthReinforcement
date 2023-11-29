from src.midas_loader import Midas
from tracker import Tracker
from features import Hungarian_Matching
import os

def main():
    SEQUENCES_FOLDER = 'data/VisDrone2019-SOT-train/sequences'
    SEQUENCE_FOLDER = 'data/VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v'
    matcher = Hungarian_Matching()
    midas = Midas()
    tracker = Tracker(matcher, midas)
    # sequences = os.listdir(SEQUENCES_FOLDER)
    # for seq in sequences[:1]:
    #     tracker.track(seq, True, fps=10)
    tracker.track(SEQUENCE_FOLDER, False, fps=10, max_idx=400)
    exit(0)
    #midas.transform_img('data/test/img0000001.jpg', 'data/depth/test')
    #midas.transform_imgs_from_folder('data/test', 'data/depth/test')

if __name__ == "__main__":
    main()