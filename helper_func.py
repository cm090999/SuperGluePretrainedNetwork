from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def transformPC(velo_pts: np.array, T: np.array):
    """
    Performs the coordinate transformation given a transformation matrix
    """
    pts, crd = np.shape(velo_pts)

    # Get LiDAR points in homogeneous shape
    velo_pts_hom = np.transpose(np.c_[velo_pts[:,0:3], np.ones(pts)])
    
    # Perform coordinate transformation and transpose the array to remove homogeneous coordinate
    velo_pts_hom_tf = np.matmul(T,velo_pts_hom)
    velo_pts_tf = np.transpose(velo_pts_hom_tf)[:,0:3]

    # Add back column vector of intensities if they were included in the input
    if crd >= 4:
        intens = velo_pts[:,-1]
        velo_pts_tf = np.c_[velo_pts_tf,intens]

    return velo_pts_tf

def veloToDepthImage(K: np.array, velo_pts: np.array, image: np.array, T = np.identity(4)):
    """
    A function that maps the points of the LiDAR point cloud to an image plane, given a calibration matrix and a transformation matrix. The image has 2 channels, [0] with depth information, [1] with intensity information.
    If no intensity chanel is given, the image has 1 channel (depth)
    """
    # Perform the coordinate transformation on the point cloud
    velo_pts_tf = transformPC(velo_pts,T)

    # Remove all points with z <= 0
    velo_pts_tf = velo_pts_tf[velo_pts_tf[:, 2] > 1e-10]

    # Initialize empty depth image
    pts, crd = np.shape(velo_pts_tf)
    if np.ndim(image) == 3:
        img_hpx, img_wpx, img_c = np.shape(image)
    if np.ndim(image) == 2:
        img_hpx, img_wpx = np.shape(image)

    chann_dep = 1
    if crd >= 4:
        chann_dep = 2

    # Initialize Depth image with 0
    depth_image = np.zeros((img_hpx,img_wpx,chann_dep))

    # get [u,v] from all point cloud points ([u,v] = K*Eye*PC)
    eyetmp = np.eye(3)
    eye = np.c_[eyetmp,np.zeros((3,1))]
    pc_img_coord_tmp = np.matmul(K,eye)
    pc_img_coord = np.matmul(pc_img_coord_tmp, np.transpose(np.c_[velo_pts_tf[:,0:3],np.zeros((pts,1))]) )
    pc_img_coord = np.transpose(pc_img_coord)
    pc_img_coord[:,0] /= pc_img_coord[:,2]
    pc_img_coord[:,1] /= pc_img_coord[:,2]
    pc_img_coord = pc_img_coord[:,0:2]

    # Get depth value of all points and attach to image coordinates array
    # attach intensity value if given
    depth_vec = np.zeros((pts,1))
    depth_vec[:,0] = np.sqrt( velo_pts_tf[:,0]**2 + velo_pts_tf[:,1]**2 + velo_pts_tf[:,2]**2 )
    pc_img_coord = np.c_[pc_img_coord,depth_vec]
    if crd >= 4:
        pc_img_coord = np.c_[pc_img_coord,velo_pts_tf[:,3]]

    # Remove points outside of image
    pc_img_coord = pc_img_coord[pc_img_coord[:,0] >= 0]
    pc_img_coord = pc_img_coord[pc_img_coord[:,1] >= 0]
    pc_img_coord = pc_img_coord[pc_img_coord[:,0] <= img_wpx]
    pc_img_coord = pc_img_coord[pc_img_coord[:,1] <= img_hpx]

    # Get pixel coordinates of all points, i.e. round coordinates
    pc_px_coord = (pc_img_coord[:,0:2]).astype(int)

    # Get indices of all duplicate coordinates
    pc_px_coord_flattened = pc_px_coord[:,0] + pc_px_coord[:,1] * img_wpx
    _, idcs = np.unique(pc_px_coord_flattened, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(pc_px_coord_flattened)), idcs)

    # Fill depth image matrix
    depth_image[pc_px_coord[:, 1], pc_px_coord[:, 0],:] = pc_img_coord[:, 2:]
    for i in range(len(duplicate_indices)):
        coord = pc_px_coord[duplicate_indices[i],:]
        if depth_image[coord[1],coord[0],0] > pc_img_coord[duplicate_indices[i],2]:
            depth_image[coord[1],coord[0],:] = pc_img_coord[duplicate_indices[i],2:]

    return depth_image

def rtvec_to_matrix(rvec=(0,0,0), tvec=(0,0,0)):
    "Convert rotation vector and translation vector to 4x4 matrix"
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)

    T = np.eye(4)
    (R, jac) = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T

def matrix_to_rtvec(matrix):
    "Convert 4x4 matrix to rotation vector and translation vector"
    (rvec, jac) = cv2.Rodrigues(matrix[:3, :3])
    tvec = matrix[:3, 3]
    return rvec, tvec

def plotOverlay(rgb, lidar, ax = None, color_map = 'jet', size_scale = 800, **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    # Display RGB
    ax.imshow(rgb)

    # Normalize Depth Images
    depth1_scaled = (lidar[:,:,0]-np.min(lidar[:,:,0]))/np.max(lidar[:,:,0])*255

    # Get the indices of non-zero elements in the depth map
    points1 = np.nonzero(lidar[:,:,0])

    # Plot depth points with scatter
    ax.scatter(points1[1], points1[0], s=depth1_scaled[points1]/size_scale, c=depth1_scaled[points1], cmap=color_map, alpha=0.99)
   
    return ax