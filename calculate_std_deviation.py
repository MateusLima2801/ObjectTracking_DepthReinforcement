import threading
from src.matchers.depth_distribution_matcher import Depth_Distribution_KL_Matcher
from src.deviation_calculator import *
import os
import json
from src.utils import file_separator
from queue import Queue
from src.job_workers import JobWorkers

SOURCE_FOLDER = os.path.join('data','VisDrone2019-MOT-test-dev')
DEPTH_SEQUENCE_FOLDER = os.path.join('data', 'depth_track')
midas = Midas()
calcs: list[Deviation_Calculator] = [ Position_Deviation_Calculator(SOURCE_FOLDER),
                                      Feature_Deviation_Calculator(SOURCE_FOLDER),
                                      Depth_Deviation_Calculator(SOURCE_FOLDER, midas),
                                      Shape_Deviation_Calculator(SOURCE_FOLDER),
                                      Depth_Distribution_Deviation_Calculator(SOURCE_FOLDER, DEPTH_SEQUENCE_FOLDER,
                                                                              Depth_Distribution_KL_Matcher(),
                                                                              midas, 100, False)]
metrics = ["position","feature","depth","shape", "depth-distribution-KL"]
deviation_file = os.path.join("data","standard_deviations.json")
sequences = [ seq.split('.')[0].split(file_separator())[-1] for seq in os.listdir(os.path.join(calcs[0].source_folder, calcs[0].annotations))]
content: dict[dict[str,float]]
test_sequences = ['uav0000009_03358_v','uav0000120_04775_v','uav0000077_00720_v', 'uav0000201_00000_v', 'uav0000297_02761_v', 'uav0000119_02301_v']
lock = threading.Lock()
seq_queue = Queue()

def iterate_a_sequence(seq: str, args):
    content, metrics, deviation_file, bar, test_sequences = args
    std = calc.calculate_for_a_sequence(seq)
    print(f'Sequence {seq} - Standard Deviation: {std}')
    lock.acquire()
    content['standard-deviations'][seq][metrics[i]] = std
    f = open(deviation_file, "w")
    json.dump(content, f)
    f.close()
    bar.next()
    lock.release()
    if seq not in test_sequences:
            depth_file = os.path.join('data', 'depth_track', f'{seq}.tar.gz')
            if os.path.isfile(depth_file):
                os.remove(depth_file)

for seq in test_sequences:
    sequences.remove(seq)
sequences = test_sequences + sequences

if not os.path.isfile(deviation_file):
    content = {"standard-deviations": {},
               "mean-standard-deviations": {}}
    for seq in sequences:
        content['standard-deviations'][seq] = {}
else:
    f = open(deviation_file)
    content = json.load(f)
    f.close()

for i, calc in enumerate(calcs):
    if metrics[i] not in content['mean-standard-deviations'].keys():
        content['mean-standard-deviations'][metrics[i]] = None
    if content['mean-standard-deviations'][metrics[i]] == None:
        sum = 0
        bar = Bar(f"Processing std for {metrics[i]}...", max = len(sequences))
        for seq in sequences:
            if metrics[i] not in content['standard-deviations'][seq].keys():
                content['standard-deviations'][seq][metrics[i]] = None
            if content['standard-deviations'][seq][metrics[i]] == None:
                seq_queue.put(seq)
            else: bar.next()
        job = JobWorkers(seq_queue, iterate_a_sequence, 1, False, content, metrics, deviation_file, bar, test_sequences)
        for seq in sequences:
            sum += content['standard-deviations'][seq][metrics[i]]
        mean = sum /len(sequences)
        print(f'Mean Standard Deviation: {mean}')
        content['mean-standard-deviations'][metrics[i]] = mean
        f = open(deviation_file, "w")
        json.dump(content, f)
        f.close()