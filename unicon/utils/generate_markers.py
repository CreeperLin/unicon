import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
# import pandas as pd
# %matplotlib nbagg

dict_list = [
    # aruco.DICT_4X4_100,
    # aruco.DICT_5X5_100,
    # aruco.DICT_6X6_100,
    # aruco.DICT_7X7_100,
    aruco.DICT_APRILTAG_36h11,
]

NX = 3
NY = 4
NPAGE = 2
STARTID = 0
POSTFIX = 'pdf'

for idx, marker in enumerate(dict_list):
    for page in range(NPAGE):
        aruco_dict = aruco.getPredefinedDictionary(marker)
        fig = plt.figure()
        
        fig.set_size_inches(8.27, 11.69)
        
        for i in range(1, NX*NY+1):
            ax = fig.add_subplot(NY, NX, i)
            print("table id", STARTID+i+page*NX*NY)
            img = aruco.generateImageMarker(aruco_dict, STARTID+i+page*NX*NY, 700)
            # print("img", img.shape)
            plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
            ax.axis("off")

        # plt.savefig(f"resources/markers_36h11_{page}.pdf")
        plt.savefig(f"sensors/markers_36h11_table_{page}.{POSTFIX}")
        # set size of a4
        # fig.set_size_inches(11.69, 8.27)
        plt.close()

# plt.show()