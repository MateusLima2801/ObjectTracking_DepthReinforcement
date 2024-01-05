from detector import Detector
from src.midas_loader import Midas
from tracker import Tracker
from src.features import Hungarian_Matching

def main():
    SEQUENCES_FOLDER = 'data/VisDrone2019-SOT-train/sequences'
    SEQUENCE_FOLDER = 'data/VisDrone2019-MOT-test-dev/sequences/uav0000297_02761_v'
    GROUND_TRUTH_FILEPATH ='data/VisDrone2019-MOT-test-dev/annotations/uav0000297_02761_v.txt'
    SEQUENCE_FOLDER = 'data\\VisDrone2019-MOT-test-dev\\sequences\\uav0000297_02761_v'
    GROUND_TRUTH_FILEPATH ='data\\VisDrone2019-MOT-test-dev\\annotations\\uav0000297_02761_v.txt'
    # [FEATURE, POSITION, DEPTH, DISPLACEMENT]
    weights = [[1/3,1/3,0,1/3]] #[[1, 0, 0], [0.5,0.5,0],[0.5, 0, 0.5], [1/3,1/3,1/3]]
    supp = [True]#[False, True]
    matcher = Hungarian_Matching()
    if weights[0][2] == 0: midas = None
    else: midas = Midas()
    detector = Detector()
    tracker = Tracker(matcher, midas, detector)
    STD_DEVIATIONS = [4.080301076630467,4.1468104706547075,0.4823281584040535,1]

    # [FEATURE, POSITION, DEPTH]
    weights = [[1/3,1/3,1/3]] #[[1, 0, 0], [0.5,0.5,0],[0.5, 0, 0.5], [1/3,1/3,1/3]]
    SEQUENCE_FOLDER = 'data\\VisDrone2019-MOT-test-dev\\sequences\\uav0000297_02761_v'
    GROUND_TRUTH_FILEPATH ='data\\VisDrone2019-MOT-test-dev\\annotations\\uav0000297_02761_v.txt'
    # [FEATURE, POSITION, DEPTH, DISPLACEMENT]
    weights = [[1/3,1/3,0,1/3]] #[[1, 0, 0], [0.5,0.5,0],[0.5, 0, 0.5], [1/3,1/3,1/3]]
    supp = [True]#[False, True]
    matcher = Hungarian_Matching()
    if weights[0][2] == 0: midas = None
    else: midas = Midas()
    detector = Detector()
    tracker = Tracker(matcher, midas, detector)
    STD_DEVIATIONS = [4.080301076630467,4.1468104706547075,0.4823281584040535,1]

    

    for s in supp:
        for w in weights:
            tracker.track(SEQUENCE_FOLDER, fps=10, max_idx=None,delete_imgs=True,weights=w,ground_truth_filepath=GROUND_TRUTH_FILEPATH, conf=0.35, suppression=s, std_deviations = STD_DEVIATIONS)

if __name__ == "__main__":
    main()