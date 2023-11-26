from ultralytics import YOLO
from deprecated.midas_loader import Midas

class Segmentor():
    def __init__(self):
        self.midas = Midas()

    def segment(self, source: str,  conf: float=0.3):
        model = YOLO("yolov8n-seg.pt") # for detection
        prediction = model.predict(source=source, save=True, conf = conf, save_txt=True, save_conf=True, hide_labels=True) 

        #model.export(format="onnx")
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
            labels = self.cast_list(labels, str)
            for label in labels:
                line = ' '.join(label) + '\n'
                f.write(line)

    def cast_list(self, test_list, data_type):
        return list(map(data_type, test_list))

seg = Segmentor()
seg.segment('data/test/img0000001.jpg', conf=0.2)
#det.retrieve_depth('data/test/dog.jpg', 'runs/detect/predict3/labels/dog.')