from src.deviation_calculator import *
import os
import json

SOURCE_FOLDER = 'data/VisDrone2019-MOT-test-dev'
calcs: list[DeviationCalculator] = [ PositionDeviationCalculator(SOURCE_FOLDER), FeatureDeviationCalculator(SOURCE_FOLDER),DepthDeviationCalculator(SOURCE_FOLDER)]
metrics = ["position","feature","depth"]
deviation_file = "data/standard_deviations.json"
sequences = [ seq.split('.')[0].split('/')[-1] for seq in os.listdir(os.path.join(calcs[0].source_folder, calcs[0].annotations))]

if not os.path.isfile(deviation_file):
    content = {"standard-deviations": {},
               "mean-standard-deviations": {"position": None, "depth": None, "feature": None}}
    for seq in sequences:
        content['standard-deviations'][seq] = {"position": None, "depth": None, "feature": None}
else:
    f = open(deviation_file)
    content = json.load(f)
    f.close()

for i, calc in enumerate(calcs):
    if content['mean-standard-deviations'][metrics[i]] == None:
        sum = 0
        for seq in sequences:
            if content['standard-deviations'][seq][metrics[i]] == None:
                std = calc.calculate_for_a_sequence(seq)
                print(f'Sequence {seq} - Standard Deviation: {std}')
                content['standard-deviations'][seq][metrics[i]] = std
                f = open(deviation_file, "w")
                json.dump(content, f)
                f.close()
            sum += content['standard-deviations'][seq][metrics[i]]
        mean = sum /len(seq)
        print(f'Mean Standard Deviation: {mean}')
        content['mean-standard-deviations'][metrics[i]] = mean
        f = open(deviation_file, "w")
        json.dump(content, f)
        f.close()