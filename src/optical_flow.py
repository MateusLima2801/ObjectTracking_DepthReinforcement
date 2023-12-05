import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import abc
import sys
import torch
sys.path.append('/home/mateus/Desktop/ObjectTracking_DepthReinforcements')

from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT
from src.utils import *

class AbstractOptFlow(abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def push_forward(self, img1, img2, pts):
        pass

class DenseOptFlow(AbstractOptFlow, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def flow(self, img1, img2):
        """Compute the dense optical flow from img1 to img2."""
        pass

    def push_forward(self, img1, img2, pts):
        u_img, v_img = self.flow(img1, img2).numpy()
        ppts = np.copy(pts) # projected points
        
        u_p, v_p = u_img[ppts[:, 1], ppts[:, 0]], v_img[ppts[:, 1], ppts[:, 0]]
        v_p = -v_p # /!\ Orientation of the axis is than that of flow
        ppts[:,0] += cast_list(u_p, int)
        ppts[:,1] += cast_list(v_p, int)

        return ppts

class RAFTOptFlow(DenseOptFlow):
    def __init__(self, tag="raft-things.pth", niters =15, device="cpu") -> None:
        args = dotdict({
            "model": f'RAFT/models/{tag}',
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        })

        self.niters = niters
        self.device = device
        self.model = RAFT(args)
        self.model.to(self.device)

    def flow(self, img1, img2):
        load = lambda x: torch.tensor(x, 
                                        device=self.device, 
                                        dtype=torch.float32
                                        ).permute(2, 0, 1)[None]
        img1, img2 = load(img1), load(img2)

        padder = InputPadder(img1.shape)
        img1, img2= padder.pad(img1, img2)

        with torch.inference_mode():
            _, flow_up = self.model(img1, img2, iters=self.niters, test_mode=True)
        
        flow_up = padder.unpad(flow_up)
        return flow_up[0].cpu()

    

# turn_imgs_into_video('data/optical_flow', 'test_of', 'jpeg')
# cap = cv.VideoCapture('data/track_video/test_of.mp4')

# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# # Create some random colors
# color = np.random.randint(0, 255, (100, 3))

# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# name = 0
# while(1):
#     ret, frame = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         break

#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     # Select good points
#     if p1 is not None:
#         good_new = p1[st==1]
#         good_old = p0[st==1]

#     # draw the tracks
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
#         frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
#     img = cv.add(frame, mask)

#     plt.imsave(f'data/optical_flow/img{name}.jpeg', img)
#     # k = cv.waitKey(30) & 0xff
#     # if k == 27:
#     #     break

#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)
#     name+=1

# turn_imgs_into_video('data/optical_flow', 'test_of', 'jpeg')
