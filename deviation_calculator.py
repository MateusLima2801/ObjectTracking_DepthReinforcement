
import math
import os
import numpy as np
import json
from src.job_workers import JobWorkers
from src.frame import Frame
from src.midas_loader import Midas
from src.bounding_box import BoundingBox
import src.utils as utils
from progress.bar import Bar
from queue import Queue

class DeviationCalculator():
    annotations = "annotations"
    sequences = "sequences"

    def __init__(self, source_folder: str):
        self.source_folder = source_folder

    def calculate(self) -> float:
        sequences = os.listdir(os.path.join(self.source_folder, self.annotations))
        sum = 0
        for seq in sequences:
            seq_name = seq.split('.')[0].split('/')[-1]
            std = self.calculate_for_a_sequence(seq_name)
            sum += std
            print(f'Sequence {seq_name} - Standard Deviation: {std}')
        return sum / len(sequences)

    def calculate_for_a_sequence(self, sequence: str) -> float:
        raise NotImplementedError
    
    def create_bounding_box(self, line: str) -> BoundingBox  | float:
        info = utils.cast_list(line.replace('\n', '').split(','), int)
        return BoundingBox(int((info[2]+info[4])/2), int((info[3]+info[5])/2), info[4], info[5], 1, id=info[1]), info[0]

    def read_sequence_annotations(self, sequence: str):
        annotations_file_path = os.path.join(self.source_folder, self.annotations, sequence + '.txt')
        f = open(annotations_file_path, "r")
        lines = f.readlines()
        f.close()
        return lines
    
class PositionDeviationCalculator(DeviationCalculator):

    def calculate_for_a_sequence(self, sequence: str) -> float:
        lines = self.read_sequence_annotations(sequence)

        square_sum = 0
        counter = 0
        next_bb, _ = self.create_bounding_box(lines[0])
        for i in range(len(lines[:-1])):
            bb = next_bb
            next_bb, _ = self.create_bounding_box(lines[i+1])
            if bb.id == next_bb.id:
                square_sum += (bb.x - next_bb.x)**2 + (bb.y - next_bb.y)**2 
                counter +=1

        return math.sqrt(square_sum / counter)
    
class DepthDeviationCalculator(DeviationCalculator):
    depth_sequences = "depth_sequences"
    #store depth arrays in json
    def __init__(self, source_folder: str):
        super().__init__(source_folder)
        self.midas = Midas()
        self.depth_folder = os.path.join(self.source_folder,self.depth_sequences)

    def calculate_for_a_sequence(self, sequence: str) -> float:
        lines = self.read_sequence_annotations(sequence)

        frames = {}
        os.makedirs(os.path.join(self.depth_folder,sequence),exist_ok = True)
        img_source = os.path.join(self.source_folder, self.sequences, sequence)
        img_names = np.array(utils.get_filenames_from(img_source, 'jpg'))
        bar = Bar("Loading depth arrays...", max=len(img_names))
        q = Queue()
        for i in img_names:
            q.put(i)

        j = JobWorkers(q, self.iterate_depth_retrieval, 2,True, img_source, frames, bar, sequence)

        square_sum = 0
        counter = 0
        next_bb, f_id = self.create_bounding_box(lines[0])
        next_bb.depth = self.get_centroid_depth(f_id, frames, next_bb)
        for i in range(len(lines[:-1])):
            bb = next_bb
            next_bb, f_id = self.create_bounding_box(lines[i+1])
            next_bb.depth = self.get_centroid_depth(f_id, frames, next_bb)
            if bb.id == next_bb.id:
                square_sum += (bb.depth - next_bb.depth)**2 
                counter +=1

        return math.sqrt(square_sum / counter)
    
    def get_centroid_depth(self, frame_id: int , frames: dict[int,Frame], bb: BoundingBox) -> float:
        depth_array = frames[frame_id].depth_array
        return float(depth_array[Frame.interpol(bb.y,len(depth_array)), Frame.interpol(bb.x, len(depth_array[0]))])

    def iterate_depth_retrieval(self, name: str, args):
        img_source, frames, bar, sequence = args
        json_name = os.path.join(self.depth_folder, sequence, '.'.join([name.split('.')[0], 'json']))
        load_midas = False
        if os.path.isfile(json_name):
            try: depth_array = self.read_json(json_name)
            except: load_midas = True
        else: load_midas = True
        
        if load_midas:
            img_path = os.path.join(img_source, name)
            img = utils.get_img_from_file(img_path)
            depth_array = self.midas.get_depth_array(img)
            self.write_json(json_name, depth_array)
        id = utils.get_number_from_filename(name)
        frames[id] = Frame(id,None, None, depth_array)
        bar.next()
    
    def write_json(self, filepath, depth_array:np.ndarray):
        content = {'depth_array': depth_array.tolist()}
        f = open(filepath, "w")
        json.dump(content, f)
        f.close()

    def read_json(self, filepath):
        f = open(filepath)
        data = json.load(f)
        f.close()
        return np.array(data['depth_array'])

calc = DepthDeviationCalculator("data/VisDrone2019-MOT-test-dev")
std = calc.calculate()
print(std)

