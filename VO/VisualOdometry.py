# based on: https://github.com/uoip/monoVO-python

import numpy as np
import cv2
import logging
import time

class VisualOdometry(object):
    """
    A simple frame by frame visual odometry
    """

    def __init__(self, detector, matcher, cam, reset_idx = None):
        """
        :param detector: a feature detector can detect keypoints their descriptors
        :param matcher: a keypoints matcher matching keypoints between two frames
        :param cam: camera parameters
        """
        # feature detector and keypoints matcher
        self.detector = detector
        self.matcher = matcher

        # camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # frame index counter
        self.index = 0

        # keypoints and descriptors
        self.kptdescs = {}

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

        # custom logger
        self.logger = self.setup_logger()

        # vo accuracy params
        self.inliers_ = []
        self.thetaY_ =  []

        if reset_idx:
            self.reset_idx = reset_idx

    

    def setup_logger(self):
        logger = logging.getLogger('VisualOdometry')
        logger.setLevel(logging.INFO)  # Step 4: Set the logger level

        file_handler = logging.FileHandler('visual_odometry.log')  # Log to a file
        console_handler = logging.StreamHandler()  # Log to the console

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

        # # Example usage
        # inlier_cnt = 100  # Example variable
        # R = "some_matrix"  # Example variable
        # logger.info(f"INLIER_CNT: {inlier_cnt} ")
        # logger.info(f"THETA_Y: {R}")  # Assuming self.thetaY(R) returns 'R' for this example


    def thetaY(self, R): 
        theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        return np.degrees(theta)

    def get_inliers(self):
        return self.inliers_

    def get_thetaYs(self):
        return self.thetaY_
    
    

    def update(self, image, absolute_scale=1):
        """
        update a new image to visual odometry, and compute the pose
        :param image: input image
        :param absolute_scale: the absolute scale between current frame and last frame
        :return: R and t of current frame
        """
        # logging.warning(f"[VO IDX]: {self.index}")
        # resetting 
        if self.reset_idx > 0:
            if self.index > 0 and self.index % self.reset_idx == 0:
                self.index = 0
                logging.warning("=======================")
                logging.warning(f"[VisualOdometry] reset")
                logging.warning("=======================")
                time.sleep(2)
        
        kptdesc = self.detector(image)
        
        # first frame
        if self.index == 0:
            # save keypoints and descriptors
            self.kptdescs["cur"] = kptdesc

            # start point
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
        else:
            # update keypoints and descriptors
            self.kptdescs["cur"] = kptdesc

            # match keypoints
            matches = self.matcher(self.kptdescs)

            # compute relative R,t between ref and cur frame
            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                           focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            inlier_cnt, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)
            
            # logging.info(f"INLIER_CNT: {inlier_cnt} ")
            # logging.info(f"THETA_Y: {self.thetaY(R)}")
            self.logger.info(f"INLIER_CNT: {inlier_cnt} ")
            self.logger.info(f"THETA_Y: {self.thetaY(R)}")
            
            # get absolute pose based on absolute_scale
            # if (absolute_scale > 0.1):
            #     self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            #     self.cur_R = R.dot(self.cur_R)
            self.inliers_.append(inlier_cnt)
            self.thetaY_.append(self.thetaY(R))
            
            # flag conditions
            inlier_cutoff = 50
            x, y,  z = t[0], t[1], t[2]

            flag = False
            flag = flag or (inlier_cnt > inlier_cutoff)
            flag = flag or (abs(z) > abs(x))
            flag = flag or (abs(z) > abs(y))    

            if (flag):
                # self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_t = self.cur_t + 1.0 * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)
            else:
                logging.error("=======================")
                logging.error(f"FLAG CONDITION NOT MET!")  
                logging.error("=======================")
                time.sleep(2)

        self.kptdescs["ref"] = self.kptdescs["cur"]

        self.index += 1
        return self.cur_R, self.cur_t


class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose

        scale = 1.0
        if self.count != 0:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) * (self.cur_pose[2, 3] - self.prev_pose[2, 3]))

        self.count += 1
        self.prev_pose = self.cur_pose
        return scale


if __name__ == "__main__":
    from DataLoader.KITTILoader import KITTILoader
    from Detectors.HandcraftDetector import HandcraftDetector
    from Matchers.FrameByFrameMatcher import FrameByFrameMatcher

    loader = KITTILoader()
    detector = HandcraftDetector({"type": "SIFT"})
    matcher = FrameByFrameMatcher({"type": "FLANN"})
    absscale = AbosluteScaleComputer()

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        R, t = vo.update(img, absscale.update(gt_pose))
