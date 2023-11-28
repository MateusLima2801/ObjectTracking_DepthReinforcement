from ultralytics import YOLO
from src.utils import *

class Detector():
    # def __init__(self):
    #     self.model_label = {"detection": "yolov8n.pt", "segmentation": "yolov8n-seg.pt"}

    @staticmethod
    def detect(source: str,  conf: float=0.3):
        model = YOLO("yolov8n.pt") # for detection

        ##Predict Method Takes all the parameters of the Command Line Interface
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, syave_txt=True)
        #model.predict(source='data/demo.mp4', save=True, conf=0.5, save_txt=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.8, save_txt=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, save_crop=True)
        #model.predict(source='data/image1.jpg', save=True, conf=0.5, hide_labels=True, hide_conf = True)
        prediction = model.predict(source=source, conf = conf, save = True, save_txt=True, save_conf=True)
        #model.export(format="onnx")

    @staticmethod
    def detect_and_read_labels(source: str, conf: float=0.3) -> dict:
        delete_folder('runs/detect')
        Detector.detect(source, conf)
        return Detector.read_labels_from('runs/detect/predict')
    
    @staticmethod
    def read_labels_from(source) -> dict:
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

# det = Detector()
Detector.detect('data/VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v/0000043.jpg', conf=0.1)
# det.detect('data/test/img0000014.jpg', conf=0.2)
# print(r)
# labels = Detector.read_labels_from('runs/detect/predict5')
# print(labels)
#det.retrieve_depth('data/test/dog.jpg', 'runs/detect/predict3/labels/dog.')