import cv2
import torch
import matplotlib.pyplot as plt
import os
from progress.bar import Bar
import json

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
        if os.path.isfile(input_path) == False: 
            raise Exception("Wrong input format")
        filename = os.path.split(input_path)[1]
        output_img_folder = os.path.join(output_folder,'img')
        output_array_folder = os.path.join(output_folder,'array')
        output_img_path = os.path.join(output_img_folder,filename)
        json_filename = f'{filename.split(".")[0]}.json '
        output_array_path = os.path.join(output_array_folder, json_filename)

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

        depth_array = prediction.cpu().numpy().tolist()
        data = { "depth_matrix": depth_array}
        os.makedirs(output_array_folder, exist_ok = True)
        # Convert the Python array to a JSON-serializable format
        json_obj = json.dumps(data)

        # Save the JSON data to a file
        with open(output_array_path, 'w') as json_file:
            json_file.write(json_obj)

        os.makedirs(output_img_folder, exist_ok = True)
        plt.imsave(output_img_path, depth_array)
    
    def transform_imgs_from_folder(self, input_folder: str, output_folder: str):
        img_files = os.listdir(input_folder)
        with Bar(max = len(img_files), suffix='%(percent).1f%% - %(eta)ds') as bar:
            for img_file in img_files:
                img_path = os.path.join(input_folder, img_file)
                self.transform_img(img_path, output_folder)
                bar.next()

    