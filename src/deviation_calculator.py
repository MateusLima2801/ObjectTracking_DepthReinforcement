
import math
import os
import shutil
import numpy as np
import json
from src.features import Feature_Matcher
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
        mean = sum / len(sequences)
        print(f'Mean Standard Deviation: {mean}')
        return mean

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
        bar = Bar("Processing lines...", max=len(lines)-1)
        for i in range(len(lines[:-1])):
            bb = next_bb
            next_bb, _ = self.create_bounding_box(lines[i+1])
            if bb.id == next_bb.id:
                square_sum += (bb.x - next_bb.x)**2 + (bb.y - next_bb.y)**2 
                counter +=1
            bar.next()

        return math.sqrt(square_sum / counter)
    
class DepthDeviationCalculator(DeviationCalculator):
    depth_annotations = "depth_annotations"
    def __init__(self, source_folder: str):
        super().__init__(source_folder)
        self.midas = Midas()
        self.depth_folder = os.path.join(self.source_folder,self.depth_annotations)

    def calculate_for_a_sequence(self, sequence: str) -> float:
        lines = self.read_sequence_annotations(sequence)
        lines_by_frame = {}

        frame_depths = {}
        os.makedirs(os.path.join(self.depth_folder,sequence),exist_ok = True)
        img_source = os.path.join(self.source_folder, self.sequences, sequence)
        img_names = np.array(utils.get_filenames_from(img_source, 'jpg'))
        bar = Bar("Loading depth arrays...", max=len(img_names))
        q = Queue()
        for i, name in enumerate(img_names):
            q.put(name)
            lines_by_frame[i+1] = []
        
        for line in lines:
            f_id = int(line.split(',')[0])
            lines_by_frame[f_id].append(line)

        j = JobWorkers(q, self.iterate_depth_retrieval,2,True, img_source, frame_depths, bar, sequence, lines_by_frame)

        square_sum = 0
        counter = 0
        next_bb, f_id = self.create_bounding_box(lines[0])
        next_bb.depth = frame_depths[f_id][str(next_bb.id)]
        bar = Bar("Processing lines...", max=len(lines)-1)
        for i in range(len(lines[:-1])):
            bb = next_bb
            next_bb, f_id = self.create_bounding_box(lines[i+1])
            next_bb.depth = next_bb.depth = frame_depths[f_id][str(next_bb.id)]
            if bb.id == next_bb.id:
                square_sum += (bb.depth - next_bb.depth)**2 
                counter +=1
            bar.next()
        shutil.rmtree(os.path.join(self.depth_folder, sequence))
        return math.sqrt(square_sum / counter)
    
    def get_centroid_depth(self, depth_array: np.ndarray, bb: BoundingBox) -> float:
        return float(depth_array[Frame.interpol(bb.y,len(depth_array)), Frame.interpol(bb.x, len(depth_array[0]))])

    def iterate_depth_retrieval(self, name: str, args):
        img_source, frame_depths, bar, sequence, lines_by_frame = args
        frame_id = utils.get_number_from_filename(name)
        json_name = os.path.join(self.depth_folder, sequence, '.'.join([name.split('.')[0], 'json']))
        load_midas = False
        if os.path.isfile(json_name):
            try: bboxes_depth = self.read_json(json_name)
            except: load_midas = True
        else: load_midas = True

        if load_midas:
            img_path = os.path.join(img_source, name)
            img = utils.get_img_from_file(img_path)
            depth_array = self.midas.get_depth_array(img)
            bboxes_depth = self.write_json(json_name, depth_array, lines_by_frame, frame_id)
        frame_depths[frame_id] = bboxes_depth
        bar.next()
    
    def write_json(self, filepath, depth_array:np.ndarray, lines_by_frame: dict, frame_id: int):
        content = {'bboxes_depth': {}}
        f_lines = lines_by_frame[frame_id]
        for f_line in f_lines:
            bb, _ = self.create_bounding_box(f_line)
            depth = self.get_centroid_depth(depth_array, bb)
            content['bboxes_depth'][str(bb.id)] = depth
        f = open(filepath, "w")
        json.dump(content, f)
        f.close()
        return content['bboxes_depth']

    def read_json(self, filepath):
        f = open(filepath)
        data = json.load(f)
        f.close()
        return data['bboxes_depth']
    

class FeatureDeviationCalculator(DeviationCalculator):
    def __init__(self, source_folder: str):
        super().__init__(source_folder)
        self.matcher = Feature_Matcher()

    def calculate_for_a_sequence(self, sequence: str):
        lines = self.read_sequence_annotations(sequence)
        square_sum = 0
        counter = 0
        next_bb, f1_id = self.create_bounding_box(lines[0])
        sec_next_bb, f2_id = self.create_bounding_box(lines[1])
        des2 = self.get_descriptors(sequence, next_bb, f1_id)
        des3 = self.get_descriptors(sequence, sec_next_bb, f2_id)
        bar = Bar("Processing lines...", max=len(lines)-2)
        for i in range(len(lines[:-2])):
            bb, f0_id, des1 = next_bb, f1_id, des2
            next_bb, f1_id, des2 = sec_next_bb, f2_id, des3
            sec_next_bb, f2_id = self.create_bounding_box(lines[i+2])
            des3 = self.get_descriptors(sequence, sec_next_bb, f2_id)
            if bb.id == next_bb.id and next_bb.id == sec_next_bb.id:
                matches1 = len(self.matcher.get_matches(des1, des2))
                matches2 = len(self.matcher.get_matches(des2, des3))
                square_sum += (matches1 - matches2)**2 
                counter +=1
            bar.next()
        return math.sqrt(square_sum / counter)
    
    def get_descriptors(self, sequence: str, bb: BoundingBox, f_id: int):
        mask = [bb.crop_mask(os.path.join(self.source_folder, self.sequences, sequence, utils.get_filename_from_number(f_id)))]
        _, des = self.matcher.detect_keypoints(mask)
        return des[0]
    
