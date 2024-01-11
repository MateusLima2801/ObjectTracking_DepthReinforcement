import os
from detector import Detector
from src.midas_loader import Midas
from tracker import Tracker
from src.features import Hungarian_Matching
from src import utils
from src.suppression import *

def main():
    SEQUENCES_FOLDER = os.path.join('data','VisDrone2019-MOT-test-dev','sequences')
    SEQUENCE = 'uav0000297_02761_v'
    SEQUENCE_FOLDER = os.path.join(SEQUENCES_FOLDER, SEQUENCE)
    # [FEATURE, POSITION, DEPTH, SHAPE]
    weights = [[1,1,0,1], [1,1,1,1]]
    supp = [ ParallelNMS ]
    matcher = Hungarian_Matching()
    midas = None
    for w in weights:
        if w[2] > 0: 
            midas = Midas()
            break
    detector = Detector()
    tracker = Tracker(matcher, midas, detector)
    STD_DEVIATIONS = [4.080301076630467,4.1468104706547075,0.4823281584040535,2.2988134815327603]
    sequences = ['uav0000077_00720_v', 'uav0000120_04775_v', 'uav0000201_00000_v', 'uav0000370_00001_v', 'uav0000119_02301_v']
    done =  []
    for sup in supp:
        for seq in sequences:
            for w in weights: 
                if (seq, w[2]) in done: continue
                ground_truth_filepath = os.path.join('data','VisDrone2019-MOT-test-dev','annotations', f'{seq}.txt')
                seq_path = os.path.join(SEQUENCES_FOLDER, seq)
                tracker.track(seq_path, fps=10, max_idx=50,delete_imgs=True,weights=w,ground_truth_filepath=ground_truth_filepath, conf=0.35, suppression=sup, std_deviations = STD_DEVIATIONS)

if __name__ == "__main__":
    main()