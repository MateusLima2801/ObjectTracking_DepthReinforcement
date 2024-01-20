import json
import cv2
import torch
import matplotlib.pyplot as plt
import os
from progress.bar import Bar
import numpy as np

class Midas:
    def __init__(self):
        self.model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

    def transform_img(self, input_path: str, output_folder: str):
        depth_array = self.get_depth_array_from_str(input_path)
        filename = os.path.split(input_path)[-1]
        output_path = os.path.join(output_folder,filename)
        os.makedirs(output_folder, exist_ok = True)
        plt.imsave(output_path, depth_array)
    
    def transform_imgs_from_folder(self, input_folder: str, output_folder: str):
        img_files = os.listdir(input_folder)
        with Bar(max = len(img_files), suffix='%(percent).1f%% - %(eta)ds') as bar:
            for img_file in img_files:
                img_path = os.path.join(input_folder, img_file)
                self.transform_img(img_path, output_folder)
                bar.next()

    def get_depth_array_from_str(self, input_path: str):
        if os.path.isfile(input_path) == False: 
            raise Exception("Wrong input format")
        
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_array = prediction.cpu().numpy()
        return depth_array
    

    def get_depth_array(self, img: np.ndarray) -> np.ndarray:
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_array = prediction.cpu().numpy()
        return depth_array
    
    def try_get_or_create_depth_array(self, img: np.ndarray, img_name: str, repository_folder: str) -> np.ndarray:
        json_path = os.path.join(repository_folder, f'{img_name.split(".")[0]}.json')

        try:
            if os.path.isfile(json_path):
                f = open(json_path)
                content = json.load(f)
                f.close()
                return np.array(content['depth-array'])
        except Exception as e:
            print(f"\nError loading {json_path}: {e}")

        depth_array = self.get_depth_array(img)
        content = { 'depth-array': depth_array.tolist()}
        f = open(json_path, "w")
        json.dump(content, f)
        f.close()
        return depth_array
            
            
    @staticmethod
    def get_depth_array_from_json(json_path):
        if os.path.isfile(json_path):
            f = open(json_path)
            content = json.load(f)
            f.close()
            return np.array(content['depth-array']), True
        else: return None, False
# m = Midas()
# m.transform_img('data\\VisDrone2019-MOT-test-dev\\sequences\\uav0000297_02761_v\\0000001.jpg', 'demo')