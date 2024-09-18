from pytorch_dragon.pytorch_dragon import sobel_filter
import cv2 as cv
import numpy as np
from timeit import default_timer as timer


class TestSobel:
    def test_sobel(self):
        path_to_ims = "./test/images"
        # CPU block
        im1 = cv.imread(f"{path_to_ims}/image1.jpg")
        # convert to grayscale
        gim1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        im2 = np.zeros_like(im1)
        im2[:, :, 0] = (
            gim1  # To display the image correctly you'll need to send the grey channel to each of the three image channels
        )
        im2[:, :, 1] = gim1
        im2[:, :, 2] = gim1

        # cuda
        print("Starting CUDA Sobel Filter calc")

        start = timer()
        output = sobel_filter(gim1)
        end = timer()
        print(f"CUDA Sobel Filter calculation took: {round(end-start, 2)}s")
        assert end - start < 10.0
