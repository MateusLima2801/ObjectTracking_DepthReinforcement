import os
import shutil
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from platform import system

def get_number_from_filename(filename:str):
    name = filename.split('.')[0][:]
    n = int(name)
    return n

def get_filename_from_number(n: int):
    return str(n).zfill(7) + ".jpg"

def get_filenames_from(imgs_folder:str , extension:str) -> list[str]:
    images = [img for img in os.listdir(imgs_folder) if img.endswith(extension)]
    images.sort(key=get_number_from_filename)
    return images

def delete_folder(folder: str):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

def cast_list(test_list, data_type):
        return list(map(data_type, test_list))

def get_img_from_file(source: str) -> np.ndarray:
    return plt.imread(source)

def turn_imgs_into_video(imgs_folder, video_filename:str, img_extension = "jpg", output_folder = os.path.join('data','track_video'), delete_imgs: bool = False, fps: float = 20.0):
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

def show_img(img: np.ndarray):
    plt.imshow(img)
    plt.show(block=True)

def file_separator():
    sys = system()
    if "Linux" in sys or "Darwin" in sys:
        return '/'
    elif "Windows" in sys:
        return '\\'
    
def interpol(n, maxi, mini = 0):
    return min(max(mini,n),maxi-1)

def compress_folder(folder: str) -> bool:
    if os.path.isdir(folder):
        cmd = f'tar -czvf {folder}.tar.gz -C {folder} .'
        os.system(wrap_cmd(cmd))
        return True
    return False

def decompress_file(file: str, output_folder: str) -> bool:
    os.makedirs(output_folder, exist_ok=True)
    if os.path.isfile(file) and file.endswith('tar.gz'):
        cmd = f'tar -xzvf {file} -C {output_folder}'
        os.system(wrap_cmd(cmd))
        return True
    
    return False

def wrap_cmd(cmd: str):
    sys = system()
    if "Linux" in sys or "Darwin" in sys:
        return cmd
    elif "Windows" in sys:
        return f'cmd /c "{cmd}"'

# Dot Dictionary
class dotdict(dict):
    """dot.notation access to dictionary attributes.
    Courtesy of https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__