import os
from MOT_evaluator import TrackingResult, MOT_Evaluator

sequences = ['uav0000077_00720_v']#os.listdir(os.path.join('data','VisDrone2019-MOT-test-dev','sequences'))
max_idx = 10
for seq in sequences:
    tr = TrackingResult(os.path.join("data", "VisDrone2019-MOT-test-dev","annotations", f"{seq}.txt"), os.path.join('data','VisDrone2019-MOT-test-dev','sequences', seq))
    tr.generate_video(os.path.join("data","groundtruth", seq), False, max_idx=max_idx)

