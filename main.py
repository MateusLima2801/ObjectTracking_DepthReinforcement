from src.deprecated.midas_loader import Midas
from detector import Detector
from tracker import Tracker

def main():
    FRAMES_FOLDER = 'data/VisDrone2019-SOT-train/sequences/uav0000080_01680_s'
    t = Tracker(FRAMES_FOLDER)
    t.track()
    exit(0)
    #midas = Midas()
    #midas.transform_img('data/test/img0000001.jpg', 'data/depth/test')
    #midas.transform_imgs_from_folder('data/test', 'data/depth/test')

if __name__ == "__main__":
    main()