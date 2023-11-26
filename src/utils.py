import os
import shutil
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import subprocess

def get_number_from_filename(filename:str):
    name = filename.split('.')[0][3:]
    n = int(name)
    return n

def get_filenames_from(imgs_folder:str , extension:str) -> list[str]:
    images = [img for img in os.listdir(imgs_folder) if img.endswith(extension)]
    images.sort(key=get_number_from_filename)
    return images

def delete_folder(folder: str):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

def cast_list(test_list, data_type):
        return list(map(data_type, test_list))

def get_img_from_file(source):
    return plt.imread(source)

def get_bbox_dimensions(img, label):
    H, W, _ = img.shape
    x,y,w,h = [int(label[1]*W), int(label[2]*H), int(label[3]*W), int(label[4]*H)]
    return x,y,w,h

def turn_imgs_into_video(imgs_folder, video_filename:str, img_extension = "jpg", output_folder = 'data/track_video', delete_imgs: bool = False, fps: float = 20.0):
    images = [img for img in os.listdir(imgs_folder) if img.endswith(f'.{img_extension}')]
    images.sort(key=get_number_from_filename)
    frame = cv.imread(os.path.join(imgs_folder, images[0]))
    height, width, layers = frame.shape

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    fourcc = 0
    video_path = os.path.join(output_folder, f'{video_filename}.avi')
    while os.path.isfile(video_path):
        path = video_path.split('.')
        info = path[0].split('_')
        info[-1] = str(int(info[-1])+1)
        video_path = '.'.join(['_'.join(info), path[1]])
    video = cv.VideoWriter(video_path, fourcc, fps, (width,height))
    
    for image in images:
        video.write(cv.imread(os.path.join(imgs_folder, image)))
    video.release()

    if delete_imgs:
        delete_files(imgs_folder, images)
        if len(os.listdir(imgs_folder)) == 0:
            os.rmdir(imgs_folder)

    convert_avi_to_mp4(video_path, fps, True)

def delete_files(folder_path, filenames):
    for name in filenames:
        path = os.path.join(folder_path, name)
        os.remove(path)

def normalize_array(array: np.ndarray):
    if np.min(array) == np.max(array): 
        return np.ones(array.shape)
    else:
        return (array - np.min(array)) / (np.max(array) - np.min(array))

# improve it, stop shell after popen
def convert_avi_to_mp4(avi_file_path: str, fps: float=20.0, delete_old_file: bool = True):
    output_name = avi_file_path.split('.')[0]
    cmd = f"ffmpeg -i '{avi_file_path}' -filter:v fps={fps} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output_name}.mp4';"
    if delete_old_file:
        cmd +=f' rm {avi_file_path};'
    os.popen(cmd)
    return True

# turn_imgs_into_video('data/track/uav0000076_00241_s_1', 'uav0000076_00241_s_1', fps=5)
# convert_avi_to_mp4('data/track_video/uav0000016_00000_s_0.avi', 20.0, True)
# while True: print('a')