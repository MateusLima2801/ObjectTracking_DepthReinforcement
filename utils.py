import os

def get_number_from_filename(filename):
    name = filename.split('.')[0]
    n = int(name)
    return n

def get_filenames_from(imgs_folder, extension):
    images = [img for img in os.listdir(imgs_folder) if img.endswith(extension)]
    images.sort(key=get_number_from_filename)
    return images

def delete_folder(folder: str):
    os.removedirs(folder)

def cast_list(test_list, data_type):
        return list(map(data_type, test_list))