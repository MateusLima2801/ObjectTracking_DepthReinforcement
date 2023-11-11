import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

def get_number_from_filename(filename):
    name = filename.split('.')[0]
    n = int(name)
    return n

def turn_imgs_into_video(imgs_folder, video_path):
    images = [img for img in os.listdir(imgs_folder) if img.endswith(".jpeg")]
    images.sort(key=get_number_from_filename)
    frame = cv.imread(os.path.join(imgs_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = 0
    fps = 20.0
    video = cv.VideoWriter(video_path, fourcc, fps, (width,height))

    for image in images:
        video.write(cv.imread(os.path.join(imgs_folder, image)))
    video.release()
    delete_files(imgs_folder, images)

def delete_files(folder_path, filenames):
    for name in filenames:
        path = os.path.join(folder_path, name)
        os.remove(path)


cap = cv.VideoCapture('data/optical_flow/road2.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
name = 0
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    plt.imsave(f'data/optical_flow/{name}.jpeg', img)
    # k = cv.waitKey(30) & 0xff
    # if k == 27:
    #     break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    name+=1

turn_imgs_into_video('data/optical_flow/', 'data/optical_flow/masked_road2.avi')

