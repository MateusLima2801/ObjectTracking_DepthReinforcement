import matplotlib.pyplot as plt
import src.utils as utils
from src.bounding_box import BoundingBox

class Frame():
    masks: list

    def __init__(self, img, read_labels):
        self.img = img
        self.bboxes = self.unnormalize_labels(read_labels)

    def crop_masks(self):
        # print(img)
        # print(labels)
        masks = []
        for bb in self.bboxes:
            w0, w1 = max(0,int(bb.x-bb.w/2)),min(self.img.shape[1], int(bb.x+bb.w/2))
            h0, h1 = max(0,int(bb.y-bb.h/2)), min(self.img.shape[0], int(bb.y+bb.h/2))
            mask = self.img[h0:h1, w0:w1]
            # self._show_mask(img, mask)
            masks.append(mask)
        self.masks = masks

    
    def show_masks(self):
        for mask in self.masks:
            plt.subplot(1,2,1)
            plt.imshow(self.img)
            plt.subplot(1,2,2)
            plt.imshow(mask)
            plt.show(block=True)
    
    def unnormalize_labels(self, read_labels):
        bboxes = []
        for i in range(len(read_labels)):
            x,y,w,h = utils.get_bbox_dimensions(self.img, read_labels[i])
            bboxes.append(BoundingBox(x,y,w,h))
        return bboxes

