from midas_loader import Midas
from detector import Detector

def main():
    midas = Midas()
    midas.transform_imgs_from_folder('data/test', 'data/depth/test')

if __name__ == "__main__":
    main()