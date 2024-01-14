import os
from src.MOT_evaluator import CLEAR_Metrics
from src.detector import Detector
from src.midas_loader import Midas
from src.tracker import Tracker
from src.hungarian_matching import Hungarian_Matching
from src import utils
from src.suppression import *

def main():
    SEQUENCES_FOLDER = os.path.join('data','VisDrone2019-MOT-test-dev','sequences')
    SEQUENCE = 'uav0000297_02761_v'
    SEQUENCE_FOLDER = os.path.join(SEQUENCES_FOLDER, SEQUENCE)
    # [FEATURE, POSITION, DEPTH, SHAPE, DEPTH_DISTRIBUTION]
    w = [1,1,0,1,0]
    supp = Confluence()
    matcher = Hungarian_Matching()
    detector = Detector()
    tracker = Tracker(matcher, None, detector)
    STD_DEVIATIONS = [4.080301076630467,4.1468104706547075,0.4823281584040535,2.2988134815327603,1]
    ground_truth_filepath = os.path.join('data','VisDrone2019-MOT-test-dev','annotations', f'{SEQUENCE}.txt')
    seq_path = os.path.join(SEQUENCES_FOLDER, SEQUENCE)
    conf_lim = [0.3125,0.375]
    conf = 0.375
    epslon = 0.2
    loss_delta = epslon * 20
    max_idx = 10
    previous_loss = None
    i = 0
    while abs(loss_delta) > epslon and i < 10:
        i+=1
        start = time()
        metrics = tracker.track(seq_path,None, max_idx=max_idx,delete_imgs=True,weights=w,ground_truth_filepath=ground_truth_filepath, conf=conf, suppression=supp, std_deviations = STD_DEVIATIONS)
        end = time()
        time_by_frame = (end-start)/max_idx
        loss = calc_loss(metrics, time_by_frame)
        print(f"{i} - Loss: {loss} - Conf: {conf}")
        if previous_loss == None: 
            conf_lim[1] = conf
        else:
            loss_delta = loss - previous_loss
            if loss_delta < 0: conf_lim[0] = conf
            else: conf_lim[1] = conf
        previous_loss = loss
        conf = (conf_lim[0]+conf_lim[1])/2    
def calc_loss(metrics: CLEAR_Metrics, time_by_frame: float):
    return (1-metrics.mot_accuracy)**2 + (metrics.mot_precision/28)**2 + (time_by_frame/5)**2

if __name__ == "__main__":
    main()