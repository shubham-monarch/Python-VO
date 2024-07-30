#! /usr/bin/env python3

import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging
import os
import fnmatch
from pathlib import Path


from utils.PinholeCamera import PinholeCamera


class KITTILoader(object):
    default_config = {
        "root_path": "../test_imgs",
        "sequence": "00",
        "start": 0
    }

    def __init__(self, input_folder,  config={}):
        # logging.warning(f"[KITTILoader] __init__")
        self.config = self.default_config
        self.config = {**self.config, **config}
        self.input_folder = input_folder
        
        self.cam = PinholeCamera(1241.0, 376.0, 1093.2768, 1093.2768, 964.989, 569.276)

        # start-idx
        self.img_id = self.config["start"]
        
        self.input_dir = self.config["root_path"] + "/sequences/" \
                                            + self.config["sequence"] + f"/{self.input_folder}"
        

        self.img_N = len([file for file in os.listdir(self.input_dir) if file.endswith('.png')]) 
        logging.info(f"self.img_N: {self.img_N}")   
        
        images_list = os.listdir(self.input_dir)
        filtered_files = fnmatch.filter(images_list, "left_*.png")
        self.sorted_images = sorted(filtered_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
       
       
    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    # def __getitem__(self, item):
    #     logging.warning(f"[ZEDLoader] __getitem__")
    #     # file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
    #     #             + "/image_0/" + str(item).zfill(6) + ".png"
    #     file_name = self.sorted_images[item] + self.sorted_images[item] 
    #     logging.info(f"file_name: {file_name}")
    #     # logging.warning(f"__getitem__ file_name: {file_name}")
    #     img = cv2.imread(file_name)
    #     (h, w) = (self.cam.height, self.cam.width)
        
    #     img = cv2.resize(img, (w, h))   
    #     return img

    def __iter__(self):
        return self

    def __next__(self):
        logging.warning(f"[ZEDLoader] __next__")        
        
        if self.img_id < self.img_N:
            file_path = Path(os.path.join(self.input_dir, self.sorted_images[self.img_id]))
            img = cv2.imread(file_path.as_posix())
            
            (h, w) = (int(self.cam.height), int(self.cam.width))
            img = cv2.resize(img, (w, h))   
            
            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]


if __name__ == "__main__":
    loader = KITTILoader()

    for img in tqdm(loader):
        cv2.putText(img, "Press any key but Esc to continue, press Esc to exit", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
        cv2.imshow("img", img)
        # press Esc to exit
        if cv2.waitKey() == 27:
            break
