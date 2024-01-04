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

    