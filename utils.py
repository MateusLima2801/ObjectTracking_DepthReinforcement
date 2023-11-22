import os
import shutil
import matplotlib.pyplot as plt
import cv2 as cv

def get_number_from_filename(filename):
    name = filename.split('.')[0][3:]
    n = int(name)
    return n

def get_filenames_from(imgs_folder, extension):
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

def turn_imgs_into_video(imgs_folder, output_folder, delete_imgs: bool = False):
    images = [img for img in os.listdir(imgs_folder) if img.endswith(".jpeg")]
    images.sort(key=get_number_from_filename)
    frame = cv.imread(os.path.join(imgs_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = 0
    fps = 20.0
    video_path = ""
    video = cv.VideoWriter(video_path, fourcc, fps, (width,height))

    for image in images:
        video.write(cv.imread(os.path.join(imgs_folder, image)))
    video.release()

    if delete_imgs:
        delete_files(imgs_folder, images)

def delete_files(folder_path, filenames):
    for name in filenames:
        path = os.path.join(folder_path, name)
        os.remove(path)