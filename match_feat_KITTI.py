import pykitti as pk
from pathlib import Path
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.cm as cm

from models.matching import Matching
from models.utils import frame2tensor, make_matching_plot, estimate_pose

if __name__ == '__main__':

    # Specify path to data
    data_path = 'Dataset_KITTI'
    date = '2011_09_26'
    drive = '0001'

    # Number of Timestamps to load
    n_time = 10

    # Load Dataset
    data_frame1 = pk.raw(data_path, date, drive, frames=range(0, n_time, 1))
    K_cal = data_frame1.calib.K_cam3

    # Extract two images, lidar datapoints and calibration data
    image0 = data_frame1.get_gray(0)[0]
    image1 = data_frame1.get_gray(1)[0]

    ### SuperGlue

    # Config Options
    nms_radius = 8 # SuperPoint Non Maximum Suppression (NMS) radius (Must be positive), default=4, type = int
    sinkhorn_iterations = 20 # Number of Sinkhorn iterations performed by SuperGlue , default=20, type=int
    match_threshold = 0.8 # SuperGlue match threshold, default=0.2, type=float
    keypoint_threshold = 0.005 # SuperPoint keypoint detector confidence threshold, default=0.005, type=float
    max_keypoints = 1024 # Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints), default=1024, type=int
    superglue = 'outdoor' # SuperGlue weights, choices={'indoor', 'outdoor'}, default='indoor'

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Tranform image0 for matching
    inp0 = frame2tensor(np.array(image0), device)

    ## Apply SuperGlue + Pose Estimation for first image with n_time next images

    # Initialize Empty lists to store results
    R_rel_list_SuperGlue = []
    t_rel_list_SuperGlue = []

    R_rel_list_cv2 = []
    t_rel_list_cv2 = []

    output_dir = Path().absolute() / 'KITTI_RES'
    output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(1,n_time):
        print("Imago 0 and Image " + str(i))
        image1 = data_frame1.get_gray(i)[0]
        
        fileName = str(i).zfill(3) + '.png'
        savePath = output_dir / fileName

        # Tranform images
        inp1 = frame2tensor(np.array(image1), device)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]

        ## Make Plot
        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']

        make_matching_plot(
            np.array(image0), np.array(image1), kpts0, kpts1, mkpts0, mkpts1, color,
            text, savePath, show_keypoints=True,
            fast_viz=True, opencv_display=True, opencv_title='Matches')

        # Pose Estimation with provided function
        pose = estimate_pose(mkpts0,mkpts1,K_cal,K_cal,1.)
        R_rel_list_SuperGlue.append(pose[0])
        t_rel_list_SuperGlue.append(pose[1])

        essential_matrix, inliers_ess = cv2.findEssentialMat(mkpts0,mkpts1,K_cal,threshold=1.0, prob=0.99999, method=cv2.RANSAC)

        # Get relative pose of the two images
        _, R_rel,t_rel, inliers_pose = cv2.recoverPose(essential_matrix,mkpts0,mkpts1,K_cal)
        R_rel_list_cv2.append(R_rel)
        t_rel_list_cv2.append(t_rel)

    resDict = {'R_rel_list_SuperGlue': R_rel_list_SuperGlue,
               't_rel_list_SuperGlue': t_rel_list_SuperGlue,
               'R_rel_list_cv2': R_rel_list_cv2,
               't_rel_list_cv2': t_rel_list_cv2}

    fout = output_dir / 'Pose.txt'
    fo = open(fout, "w")

    for k, v in resDict.items():
        fo.write(str(k) + ' >>> ' + '\n\n')
        for i in range(len(v)):
            fo.write(str(v[i]) + '\n')
            fo.write('\n')

    fo.close()