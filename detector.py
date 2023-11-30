from ultralytics import YOLO
from src.utils import *
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from PIL import Image

class Detector():
    def __init__(self):
        self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT).eval()

    def detect(self, source: str,  conf: float=0.3):
        img = Image.open(source)
        tensor = ToTensor()(img)
        batched = tensor.unsqueeze(0)
        with torch.no_grad():
            out = self.model(batched)
        first_out = out[0]
        bboxes = first_out['boxes'][first_out['scores']>=conf].tolist()
        scores = first_out['scores'][first_out['scores']>=conf].tolist()
        labels = []
        for i, bb in enumerate(bboxes):
            labels.append([bb[0], bb[1],bb[2],bb[3],scores[i]])
        return labels

    def detect_and_read_labels(self, source: str, conf: float=0.3) -> list[list]:
        bboxes = self.detect(source, conf)
        return Detector.transform_bboxes(bboxes)
    
    @staticmethod
    def transform_bboxes(bboxes: list[list[float]])->list[list]:
        for i,bb in enumerate(bboxes):
            bboxes[i] =  [int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2), int(bb[2]-bb[0]), int(bb[3]-bb[1]), bb[4]]
        return bboxes

# det = Detector()
# bboxes = det.detect_and_read_labels('data/VisDrone2019-MOT-test-dev/sequences/uav0000355_00001_v/0000040.jpg', conf=0.4)
# print(bboxes)
# det.detect('data/test/img0000014.jpg', conf=0.2)
# print(r)
# labels = Detector.read_labels_from('runs/detect/predict5')
# print(labels)
#det.retrieve_depth('data/test/dog.jpg', 'runs/detect/predict3/labels/dog.')