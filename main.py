import os
from detector import Detector
from src.midas_loader import Midas
from tracker import Tracker
from src.features import Hungarian_Matching

def main():
    SEQUENCES_FOLDER = os.path.join('data','VisDrone2019-MOT-test-dev','sequences')
    SEQUENCE = 'uav0000297_02761_v'
    SEQUENCE_FOLDER = os.path.join(SEQUENCES_FOLDER, SEQUENCE)
    GROUND_TRUTH_FILEPATH = os.path.join('data','VisDrone2019-MOT-test-dev','annotations', f'{SEQUENCE}.txt')
    # [FEATURE, POSITION, DEPTH, SHAPE]
    weights = [[1,1,0,1]] #[[1, 0, 0], [0.5,0.5,0],[0.5, 0, 0.5], [1/3,1/3,1/3]]
    supp = [True]#[False, True]
    matcher = Hungarian_Matching()
    midas = None
    for w in weights:
        if w[2] > 0: 
            midas = Midas()
            break
    detector = Detector()
    tracker = Tracker(matcher, midas, detector)
    STD_DEVIATIONS = [4.080301076630467,4.1468104706547075,0.4823281584040535,2.2988134815327603]

    for s in supp:
        for w in weights:
            tracker.track(SEQUENCE_FOLDER, fps=10, max_idx=50,delete_imgs=True,weights=w,ground_truth_filepath=GROUND_TRUTH_FILEPATH, conf=0.35, suppression=s, std_deviations = STD_DEVIATIONS)

if __name__ == "__main__":
    main()