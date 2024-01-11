import os
from src.MOT_evaluator import TrackingResult, MOT_Evaluator
from progress.bar import Bar

sequences = os.listdir(os.path.join('data','VisDrone2019-MOT-test-dev','sequences'))
max_idx = 1
bar = Bar("Processing sequences...", max=len(sequences))
for seq in sequences:
    tr = TrackingResult(os.path.join("data", "VisDrone2019-MOT-test-dev","annotations", f"{seq}.txt"), os.path.join('data','VisDrone2019-MOT-test-dev','sequences', seq))
    tr.print_visual_result(os.path.join("data","groundtruth", seq), False, max_idx=max_idx, gen_video=False)
    bar.next()

