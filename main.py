#! /usr/bin/env python3

import os

import numpy as np
import cv2
import argparse
import yaml
import logging, coloredlogs
from tqdm import tqdm

from utils.tools import plot_keypoints
from utils.vis_helpers import plot_histograms
from utils.io_utils import create_folders, delete_folders

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer
import time
import matplotlib.pyplot as plt
import fnmatch
import shutil
import random
 



def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"])


class TrajPlotter(object):
    def __init__(self, reset_idx = None):
        self.errors = []
        # self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        # visualization window dims
        self.h,self.w = (800, 700)
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.frame_cnt = 0
        if reset_idx:
            self.reset_idx = reset_idx

    def reset(self):
        # logging.info("=======================")
        # logging.info(f"[TrajPlotter] reset at {self.frame_cnt} frame!")
        # logging.info("=======================")
        # time.sleep(5)
            
        # self.traj = np.zeros((600, 1000, 3), dtype=np.uint8)
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
    def update(self, est_xyz, gt_xyz = None):

        # logging.info(f"[TJ IDX]: {self.frame_cnt}")        
        if self.reset_idx:
            if self.frame_cnt > 0 and self.frame_cnt % self.reset_idx ==  0:
                self.reset()
                # time.sleep(1)
        
        x, z = est_xyz[0], est_xyz[2]
        # x, z = est_xyz[1], est_xyz[2]
        
        self.frame_cnt += 1
        
        est = np.array([x, z]).reshape(2)
        
        draw_x, draw_y = int(x) + (self.w // 2), int(z) + (self.h // 2)
        
        # draw trajectory
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        return self.traj
'''
[TO-DO]
- tune script
- error-handling
  - vineyards/front_2024-06-05-09-48-13.svo
- add image extraction for selected segments
- abstract frame sequence extraction to support different interfaces
- implement class interface
- support for multiple base folders in testing
- add folder deletion
- prepare final demo-test case folder with well-distributed dataset
- update end condition
- make testing end to end
- handling large continuos valid segment
- single / multiple svo viable patch extraction 
- add visualization functions
- visualisze + extract the viable patch lengths 
- sampling viable segments
- integrate with main.sh
'''


''''
[TESTING STATUS]
- miscellaneous [IN PROGRESS]
- vineyards * 
  - RJM -> []
  - wente-test -> []
  - gallo -> []
- blueberry
    - 1A -> []
    - 1B -> []
- apple
    - agrimacs 
    - quincy_fresh
    - rj_orchards
    - wsu_washington
- raisins
- dairy 
'''

'''
IMPORTANT SVO FILES
- front_2023-11-03-10-51-17.svo [miscellaneous]

'''


'''
[IMPORTANT SVO FILES]
- REVERSE
    - blueberry/frogmore_site1B/front_2024-02-14-12-47-32.svo
    - vineyards/gallo/2024_06_07_utc/svo_files/front_2024-06-04-13-30-56.svo
    
- CAMERA COVER CRASH / E-CRASH
    - vineyards/RJM/
        - front_2024-06-05-09-48-13.svo
        - front_2024-06-05-09-48-13.svo
'''


def write_seq_to_disk(input_dir : str, sequences : tuple, output_dir = "outputs"):
    
    input_dir_ = os.path.join("test_imgs/sequences/00/", input_dir)

    img_N = len([file for file in os.listdir(input_dir_) if file.endswith('.png')]) 
    # logging.info(f"num_images: {img_N}")   
    
    images_list = os.listdir(input_dir_)
    filtered_files = fnmatch.filter(images_list, "left_*.png")
    sorted_images = sorted(filtered_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # updating sorted images with full path
    for i, image in enumerate(sorted_images):
        image = os.path.join(input_dir_, image) 
        sorted_images[i] = image

    output_dir = os.path.join(output_dir, input_dir)
    for (st,en) in tqdm(sequences):
        output_dir_ = os.path.join(output_dir, f"{st}_{en}")
    
        # logging.warning(f"output_dir: {output_dir_}")
        
        delete_folders([output_dir_])
        create_folders([output_dir_])
        
        for i in range(st, en + 1):
            # logging.info(f"{i}: {sorted_images[i]}")
            shutil.copy(sorted_images[i], output_dir_)    
    


def run(args, INPUT_FOLDER_PATH):
    with open(args.config, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)

    loader = create_dataloader(config["dataset"], INPUT_FOLDER_PATH)
    detector = create_detector(config["detector"])
    matcher = create_matcher(config["matcher"])

    absscale = AbosluteScaleComputer()
    # traj_plotter = TrajPlotter(RESET_IDX)

    # log
    fname = args.config.split('/')[-1].split('.')[0]
    log_fopen = open("results/" + fname + ".txt", mode='a')

    logging.warning("=======================")
    logging.info(f"fname: {fname}")
    zed_camera = loader.cam
    for attr, value in zed_camera.__dict__.items():
        logging.info(f"{attr}: {value}")
    logging.warning("=======================")
                
    # vo = VisualOdometry(detector, matcher, loader.cam)
    # vo = VisualOdometry(detector, matcher, zed_camera, RESET_IDX)
    total_frames = len(loader)
    vo = VisualOdometry(detector, matcher, zed_camera, total_frames)

    
    # x = enumerate(loader)
    
    # for i, img in tqdm(enumerate(loader), total=len(loader)):
    for i, img in enumerate(loader):
        # gt_pose = loader.get_cur_pose()
        # R, t = vo.update(img, absscale.update(gt_pose))
        
        # logging.warning(f"{i} / {total_frames} img.shape: {img.shape} ")
        logging.warning(f"PROCESSING [{i} / {total_frames}] FRAME")
        
        R, t = vo.update(img)
        
        img1 = keypoints_plot(img, vo)
        # img2 = traj_plotter.update(t)

        cv2.imshow("keypoints", img1)
        # cv2.imshow("trajectory", img2)
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()
    
    logging.info("=======================")
    logging.info("VO HAS FINISHED!")
    logging.info("=======================")
    
    return vo


    # seq_list = vo.get_viable_sequences()

    # # logging.info("=======================")
    # # logging.info(f"seq_list: {seq_list}")
    # # logging.info("=======================")
    
    # plot_histograms(seq_list)

    # # seq_pairs = vo.get_sequence_pairs()

    # # logging.info("=======================")
    # # logging.info(f"seq_list: {seq_pairs}")
    # # logging.info("=======================")
    
    # write_seq_to_disk(INPUT_FOLDER_PATH, seq_pairs)

    # logging.info(f"END OF VO PIPELINE!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')    
    parser.add_argument('--config', type=str, default='params/kitti_superpoint_flannmatch.yaml',
                        help='config file')
    

    args = parser.parse_args()
    coloredlogs.install(level='INFO', force=True)
    
    # RESET_IDX = 500
    # BASE_INPUT_FOLDER = "tested"
    # SVO_FOLDER = "front_2023-11-03-10-51-17.svo/"
    # INPUT_FOLDER_PATH = f"{BASE_INPUT_FOLDER}/{SVO_FOLDER}"

    PREFIX_FOLDER ="test_imgs/sequences/00/"
    IMAGES_FOLDER = "miscellaneous"
    INPUT_FOLDER = f"{PREFIX_FOLDER}{IMAGES_FOLDER}"
    
    # number of svo folders to test
    CUTOFF_NUM_FOLDERS = 5
    
    # storing relative and abs paths
    svo_folders_abs = []
    svo_folders_rel = []
    
    for root, dirs, files in os.walk(INPUT_FOLDER, topdown=True):
        if not dirs:
            svo_folders_abs.append(root)

    random.shuffle(svo_folders_abs)

    for i, folder in enumerate(svo_folders_abs):
        if i >= CUTOFF_NUM_FOLDERS:
            break
        
        all = folder.split('/')
        relevant = all[3:]
        folder_ ='/'.join(relevant)
        
        svo_folders_rel.append(folder_)
    
    logging.info("=======================")
    logging.info("FOLLOWING SVO FOLDERS WILL BE TESTED")
    for i, folder in enumerate(svo_folders_rel):
        logging.info(f"[{i}] {folder}")
    logging.info("=======================\n")
        
    time.sleep(2)

    for i, folder in enumerate(svo_folders_rel):
        logging.error("=======================")
        logging.error(f"STARTING VO FOR [{i} / {len(svo_folders_rel)}] FOLDER!")
        logging.error("=======================")
        time.sleep(5)
        
        vo  = run(args, folder)  
        # seq_list = vo.get_viable_sequences()
        # plot_histograms(seq_list)

        # (st1, en1), (st2, en2), ...
        seq_tuples = vo.get_sequence_pairs()

        logging.info("=======================")
        logging.info("SAVING SEQUENCES TO DISK!")
        logging.info("=======================")
        
        time.sleep(1)
        
        write_seq_to_disk(folder, seq_tuples)

    logging.info("=======================")
    logging.info("DELETING THE INPUT SVO FOLDERS!")
    logging.info("=======================")
    
    delete_folders(svo_folders_abs)
        