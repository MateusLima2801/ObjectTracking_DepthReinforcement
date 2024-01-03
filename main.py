from detector import Detector
from src.midas_loader import Midas
from tracker import Tracker
from src.features import Hungarian_Matching
from src.optical_flow import RAFTOptFlow

def main():
    SEQUENCES_FOLDER = 'data/VisDrone2019-SOT-train/sequences'
    SEQUENCE_FOLDER = 'data/VisDrone2019-MOT-test-dev/sequences/uav0000355_00001_v'
    GROUND_TRUTH_FILEPATH ='data/VisDrone2019-MOT-test-dev/annotations/uav0000355_00001_v.txt'
    matcher = Hungarian_Matching()
    midas = Midas()
    detector = Detector()
    # # opf = RAFTOptFlow()
    tracker = Tracker(matcher, midas, detector, None)
    STD_DEVIATIONS = [4.080301076630467,4.1468104706547075,0.4823281584040535]

    # [FEATURE, POSITION, DEPTH]
    weights = [[1/3,1/3,1/3]] #[[1, 0, 0], [0.5,0.5,0],[0.5, 0, 0.5], [1/3,1/3,1/3]]
    supp = [True]#[False, True]

    for s in supp:
        for w in weights:
            tracker.track(SEQUENCE_FOLDER, fps=10, max_idx=50,delete_imgs=True,weights=w,ground_truth_filepath=GROUND_TRUTH_FILEPATH, conf=0.35, suppression=s, std_deviations = STD_DEVIATIONS)

if __name__ == "__main__":
    main()