
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from src.midas_loader import Midas
import src.utils as utils

# x = x_img
# y = depth[x_img,y_img]
# z = y_img

m = Midas()
img = utils.get_img_from_file('data\\VisDrone2019-MOT-test-dev\\sequences\\uav0000077_00720_v\\0000329.jpg')
depth = m.try_get_or_create_depth_array(img, 'test0003.jpg', 'data\\depth_track')
# depth, _ = Midas.get_depth_array_from_json('data\\depth_track\\test000001.json')

x = np.linspace(0,depth.shape[1]/100, depth.shape[1])
y = np.array(depth)
z = np.linspace(0,depth.shape[0]/100, depth.shape[0])
x, z = np.meshgrid(x,z)
# Plot 3D surface plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, z, y, cmap='viridis')
ax.set_xlabel('x_img')
ax.set_ylabel('depth')
ax.set_zlabel('y_img')
ax.set_title('Depth representation')
ax.legend()

plt.show(block=True)

