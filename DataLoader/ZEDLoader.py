#! /usr/bin/env python3

import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging
import os
import fnmatch


from utils.PinholeCamera import PinholeCamera


class KITTILoader(object):
    default_config = {
        "root_path": "../test_imgs",
        "sequence": "00",
        "start": 0
    }

    def __init__(self, input_folder,  config={}):
        logging.warning(f"[KITTILoader] __init__")
        # logging.warning(f"[ZEDLoader] __init__")
        self.config = self.default_config
        self.config = {**self.config, **config}
        self.input_folder = input_folder
        # logging.info("KITTI Dataset config: ")
        # logging.info(self.config)

        self.cam = PinholeCamera(1241.0, 376.0, 1093.2768, 1093.2768, 964.989, 569.276)

        # image id
        self.img_id = self.config["start"]
        # self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/sequences/" \
        #                                     + self.config["sequence"] + "/image_0/*.png"))
        
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/sequences/" \
                                            + self.config["sequence"] + f"/{self.input_folder}/*.png"))
        
        self.base_dir  =self.config["root_path"] + "/sequences/" \
                                            + self.config["sequence"] + "/image_0/"
        # logging.warning(f"self.images_path: {self.}")
        
        
        images_list = os.listdir(self.base_dir)
        # logging.warning(f"type(self.base_dir): {type(self.base_dir)}")
        # logging.warning(f"self.base_dir: {self.base_dir}")
        filtered_files = fnmatch.filter(images_list, "left_*.png")
        self.sorted_images = sorted(filtered_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        # logging.info(f"{self.sorted_images[:30]}")
        # logging.info(f"type(self.sorted_images[0]): {type(self.sorted_images[0])}")
        
        # logging.warning(f"self.img_id: {self.img_id}")
        # logging.warning(f"self.img_N: {self.img_N}")

    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    def __getitem__(self, item):
        logging.warning(f"[ZEDLoader] __getitem__")
        file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
                    + "/image_0/" + str(item).zfill(6) + ".png"
        # logging.warning(f"__getitem__ file_name: {file_name}")
        img = cv2.imread(file_name)
        (h, w) = (self.cam.height, self.cam.width)
        
        img = cv2.resize(img, (w, h))   
        return img

    def __iter__(self):
        return self

    def __next__(self):
        # logging.warning(f"[ZEDLoader] __next__")        
        # logging.warning(f"self.img_id: {self.img_id}")  
        # logging.warning(f"self.img_N: {self.img_N}")

        if self.img_id < self.img_N:
            # file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
            #             + "/image_0/" + str(self.img_id).zfill(6) + ".png"
            file_name = self.base_dir + self.sorted_images[self.img_id]
            # logging.warning(f"__next__ file_name: {file_name}") 
            img = cv2.imread(file_name)
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
