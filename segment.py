from ultralytics import YOLO
from midas_loader import Midas

class Detector():
    def __init__(self):
        self.midas = Midas()

    def detect(source: str,  conf: float=0.3):
        model = YOLO("yolov8n.pt") # for detection

        ##Predict Method Takes all the parameters of the Command Line Interface
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, save_txt=True)
        #model.predict(source='data/demo.mp4', save=True, conf=0.5, save_txt=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.8, save_txt=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, save_crop=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, hide_labels=True, hide_conf = True)
        prediction = model.predict(source=source, save=True, conf=conf, save_txt=True, save_conf=True)

        #model.export(format="onnx")

    def retrieve_depth(self,source_img: str, source_labels: str):
        array = [[0,0],[1,1]] #self.midas.get_depth_array(source_img)
        labels = read_labels(source_labels)
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

    def read_labels(self, source_labels: str) -> list:
        with open(source_labels) as f:
            lines = f.readlines()
            labels= []
            for l in lines:
                label = self.cast_list(l[:-1].split(' '), float)
                labels.append(label)
            return labels
    
    def write_labels(self, labels: list, file_path: str):
        with open(file_path, 'w') as f:
            labels = self.cast_list(labels, str)
            for label in labels:
                line = ' '.join(label) + '\n'
                f.write(line)

    def cast_list(self, test_list, data_type):
        return list(map(data_type, test_list))

det = Detector()
#det.detect('/home/mateus/Desktop/ObjectTracking_DepthReinforcements/data/test/dog.jpg')
det.retrieve_depth('data/test/dog.jpg', 'runs/detect/predict3/labels/dog.txt')