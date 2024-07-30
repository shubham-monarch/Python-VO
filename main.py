#! /usr/bin/env python3

import os

import numpy as np
import cv2
import argparse
import yaml
import logging, coloredlogs
from tqdm import tqdm

from utils.tools import plot_keypoints

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer
import time

# [TO-DO]
# - tune script
# - tests
# - sampling
# - integrate with main.sh
# - add jump / reverse / stationary state detection 
# - extract straight segments
# -filter sharp turns that go unregistered



def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"])


class TrajPlotter(object):
    def __init__(self, reset_idx = None):
        self.errors = []
        # self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        # visualization window dims
        self.h,self.w = (800, 1500)
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.frame_cnt = 0
        if reset_idx:
            self.reset_idx = reset_idx

    def reset(self):
        logging.info(f"[TrajPlotter] reset")
        # self.traj = np.zeros((600, 1000, 3), dtype=np.uint8)
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
    def update(self, est_xyz, gt_xyz = None):
        if self.reset_idx  > 0:
            if self.frame_cnt > 0 and self.frame_cnt % self.reset_idx ==  0:
                self.reset()
                time.sleep(1)
        
        x, z = est_xyz[0], est_xyz[2]
        
        self.frame_cnt += 1
        
        est = np.array([x, z]).reshape(2)
        
        # draw_x, draw_y = int(x) + 500, int(z) + 290
        draw_x, draw_y = int(x) + (self.w // 2), int(z) + (self.h // 2)
        
        logging.info(f"(x,z): ({x},{z})")
        logging.info(f"(draw_x, draw_y): ({draw_x},{draw_y})\n")

        # draw trajectory
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        # cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 2)
        # cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)
        # cv2.rectangle(self.traj, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        # cv2.rectangle(self.traj, (10, 20), (800, 80), (0, 0, 0), -1)
        
        # draw text
        # text = "[AvgError] %2.4fm" % (avg_error)
        # cv2.putText(self.traj, text, (20, 40),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        self.frame_cnt += 1
        # time.sleep(0.1)
        return self.traj


    # def update(self, est_xyz, gt_xyz):
    #     x, z = est_xyz[0], est_xyz[2]
    #     gt_x, gt_z = gt_xyz[0], gt_xyz[2]

    #     est = np.array([x, z]).reshape(2)
    #     gt = np.array([gt_x, gt_z]).reshape(2)

    #     error = np.linalg.norm(est - gt)

    #     self.errors.append(error)

    #     avg_error = np.mean(np.array(self.errors))

    #     # === drawer ==================================
    #     # each point
    #     draw_x, draw_y = int(x) + 290, int(z) + 90
    #     true_x, true_y = int(gt_x) + 290, int(gt_z) + 90

    #     # draw trajectory
    #     cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
    #     cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 2)
    #     cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)

    #     # draw text
    #     text = "[AvgError] %2.4fm" % (avg_error)
    #     cv2.putText(self.traj, text, (20, 40),
    #                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    #     return self.traj

RESET_IDX = 500
INPUT_FOLDER = "front_2024-02-13-07-52-14.svo"

def run(args):
    with open(args.config, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)

    loader = create_dataloader(config["dataset"], INPUT_FOLDER)
    detector = create_detector(config["detector"])
    matcher = create_matcher(config["matcher"])

    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter(RESET_IDX)

    # log
    fname = args.config.split('/')[-1].split('.')[0]
    log_fopen = open("results/" + fname + ".txt", mode='a')

    logging.warning(f"=======================================")
    logging.info(f"fname: {fname}")
    zed_camera = loader.cam
    for attr, value in zed_camera.__dict__.items():
        logging.info(f"{attr}: {value}")
    logging.warning(f"=======================================")
    
    # vo = VisualOdometry(detector, matcher, loader.cam)
    vo = VisualOdometry(detector, matcher, zed_camera, RESET_IDX)

    total_frames = len(loader)

    x = enumerate(loader)
    
    for i, img in tqdm(enumerate(loader), total=len(loader)):
        # gt_pose = loader.get_cur_pose()
        # R, t = vo.update(img, absscale.update(gt_pose))
        
        logging.warning(f"{i} / {total_frames} img.shape: {img.shape} ")
        
        R, t = vo.update(img)

        # === log writer ==============================
        # print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)
        logging.info(f"{i} : ({t[0, 0]}, {t[1, 0]}, {t[2, 0]}")
        # === drawer ==================================
        
        img1 = keypoints_plot(img, vo)
        # img2 = traj_plotter.update(t, gt_pose[:, 3])
        img2 = traj_plotter.update(t)

        cv2.imshow("keypoints", img1)
        cv2.imshow("trajectory", img2)
        if cv2.waitKey(10) == 27:
            break

    # cv2.imwrite("results/" + fname + '.png', img2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    # parser.add_argument('--config', type=str, default='params/kitti_superpoint_supergluematch.yaml',
    #                     help='config file')
    
    parser.add_argument('--config', type=str, default='params/kitti_superpoint_flannmatch.yaml',
                        help='config file')
    
    # parser.add_argument('--config', type=str, default='params/kitti_sift_flannmatch.yaml',
    #                     help='config file')
    
    # parser.add_argument('--logging', type=str, default='INFO',
    #                     help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')

    args = parser.parse_args()
    coloredlogs.install(level='INFO', force=True)
    # logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)
