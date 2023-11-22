from ultralytics import YOLO
from midas_loader import Midas
from utils import *

class Detector():
    def __init__(self):
        # self.midas = Midas()
        self.model_label = {"detection": "yolov8n.pt", "segmentation": "yolov8n-seg.pt"}

    def detect(self, source: str,  conf: float=0.3):
        model = YOLO("yolov8n.pt") # for detection

        ##Predict Method Takes all the parameters of the Command Line Interface
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, syave_txt=True)
        #model.predict(source='data/demo.mp4', save=True, conf=0.5, save_txt=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.8, save_txt=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, save_crop=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, hide_labels=True, hide_conf = True)
        prediction = model.predict(source=source, conf = conf, save = True, save_txt=True, hide_labels=True, save_conf=True)
        #model.export(format="onnx")

    def detect_and_read_labels(self,source: str, conf: float=0.3) -> dict:
        delete_folder('runs/detect')
        self.detect(source, conf)
        return self.read_labels_from('runs/detect/predict')
        
    def read_labels_from(self, source) -> dict:
        path = os.path.join(source, 'labels')
        files = os.listdir(path)
        labels_by_file = {}
        for file in files:
            if file.endswith('.txt'):
                labels = []
                with open(os.path.join(path, file), 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        info = l[:-1].split(' ')
                        info = cast_list(info, float)
                        labels.append(info)
                labels_by_file[file.split('.')[0]] = labels
        return labels_by_file

    def retrieve_depth(self,source_img: str, source_labels: str):
        array = [[0,0],[1,1]] #self.midas.get_depth_array(source_img)
        labels = self.read_labels(source_labels)
        for label in labels:
            print(label)
            depth = self.extract_center_depth(label[1], label[2], array)
            label.append(depth)
        self.write_labels(labels,source_labels)

    def extract_center_depth(self, x_center: float, y_center: float, array) -> float:
        x_size = len(array[0])
        y_size = len(array)
        center_coord =  ( int(x_center*x_size), int(y_center*y_size))
        return array[center_coord[0], center_coord[1]]

    
    def write_labels(self, labels: list, file_path: str):
        with open(file_path, 'w') as f:
            labels = cast_list(labels, str)
            for label in labels:
                line = ' '.join(label) + '\n'
                f.write(line)


# det = Detector()
# det.detect('data/test/img0000001.jpg', conf=0.2)
# det.detect('data/test/img0000014.jpg', conf=0.2)
# print(r)
# labels = Detector.read_labels_from('runs/detect/predict5')
# print(labels)
#det.retrieve_depth('data/test/dog.jpg', 'runs/detect/predict3/labels/dog.')