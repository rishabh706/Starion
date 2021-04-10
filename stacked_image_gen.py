
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
class Generator:

    def __init__(self):
        pass

    def cropandstack(self, im):
        splits = 4
        first = im[0 : 200 * splits, 200:400]
        second = im[0 : 200 * splits, 1473:1573]

        return np.hstack((first, second))

    def generate(self, imgPath):
        splits = 4

        final_preds = []
        image_name = os.path.basename(imgPath)
        img = cv2.imread(imgPath)
        x, y, w, h = 0, 0, 1936, 200 * splits
        img_splits = []
        for i in range(int(48 / splits)):
            im = img[y : y + h, x : x + w]

            im = self.cropandstack(im)
            img_splits.append(im)

            y = y + 200 * splits

        for i in range(4):
            img2concat = img_splits[3 * i : 3 * (i + 1)]
            im = np.hstack(img2concat)
            # H, W, _ = im.shape
            
            cv2.imwrite(f"stacked_generated_images//{image_name}", im)


if __name__ == "__main__":

    model = Generator()
    for imgPath in tqdm(glob.glob("dataset_linescan/*.bmp")):
        print(imgPath)
        model.generate(imgPath)
