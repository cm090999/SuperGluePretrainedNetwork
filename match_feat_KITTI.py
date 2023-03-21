import pykitti as pk
from pathlib import Path
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.cm as cm

from models.matching import Matching
from helper_func import matrix_to_rtvec, rtvec_to_matrix, transformPC, veloToDepthImage, plotOverlay
from models.utils import frame2tensor, make_matching_plot

if __name__ == '__main__':

    # Specify path to data
    data_path = 'Dataset_KITTI'
    date = '2011_09_26'
    drive = '0001'

    # Load Dataset
    data_frame1 = pk.raw(data_path, date, drive, frames=range(0, 10, 1))

    # Extract two images, lidar datapoints and calibration data
    image0 = data_frame1.get_gray(0)[0]
    image1 = data_frame1.get_gray(1)[0]

    velo0 = data_frame1.get_velo(0)
    velo1 = data_frame1.get_velo(1)

    velocrd0 = velo0[:,0:3]
    velocrd1 = velo1[:,0:3]

    T_cal = data_frame1.calib.T_cam3_velo
    rvec, tvec = matrix_to_rtvec(T_cal)
    K_cal = data_frame1.calib.K_cam3

    # Get projected depth maps
    depth0 = veloToDepthImage(K_cal,velo0,image1,T_cal)
    depth1 = veloToDepthImage(K_cal,velo1,image1,T_cal)

    ### SuperGlue

    # Config Options
    nms_radius = 4 # SuperPoint Non Maximum Suppression (NMS) radius (Must be positive), default=4, type = int
    sinkhorn_iterations = 20 # Number of Sinkhorn iterations performed by SuperGlue , default=20, type=int
    match_threshold = 0.2 # SuperGlue match threshold, default=0.2, type=float
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

    # Tranform images
    inp0 = frame2tensor(np.array(image0), device)
    inp1 = frame2tensor(np.array(image1), device)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp0})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]




    # # Visualize the matches.
    # color = cm.jet(mconf)
    # text = [
    #     'SuperGlue',
    #     'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    #     'Matches: {}'.format(len(mkpts0)),
    # ]

    # # Display extra parameter info.
    # k_thresh = matching.superpoint.config['keypoint_threshold']
    # m_thresh = matching.superglue.config['match_threshold']
    # small_text = [
    #     'Keypoint Threshold: {:.4f}'.format(k_thresh),
    #     'Match Threshold: {:.2f}'.format(m_thresh),
    #     'Image Pair: {}:{}'.format(stem0, stem1),
    # ]

    # make_matching_plot(
    #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
    #     text, viz_path, opt.show_keypoints,
    #     opt.fast_viz, opt.opencv_display, 'Matches', small_text)




    fundamental_matrix, inliers_fund = cv2.findFundamentalMat(np.float32(kpts0),np.float32(kpts1),method=cv2.FM_RANSAC)
    essential_matrix, inliers_ess = cv2.findEssentialMat(np.float32(kpts0),np.float32(kpts1),K_cal,method=cv2.FM_RANSAC)

    # Get relative pose of the two images
    _, R_rel,t_rel, inliers_pose = cv2.recoverPose(essential_matrix,np.float32(kpts0),np.float32(kpts1),K_cal)

    print(R_rel,t_rel)