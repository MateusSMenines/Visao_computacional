import cv2
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import ndimage, misc


def plane_sweep_gauss(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation
    with Gaussian weighted neigborhoods. """
    m,n = im_l.shape
    
    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))
    
    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))
    
    # compute mean
    ndimage.gaussian_filter(im_l,wid,0,mean_l)
    ndimage.gaussian_filter(im_r,wid,0,mean_r)
    
    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    
    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums
      ndimage.gaussian_filter(norm_l*np.roll(norm_r,displ+start),wid,0,s)  # sum nominator
      ndimage.gaussian_filter(norm_l*norm_l,wid,0,s_l)	
      ndimage.gaussian_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,0,s_r) # sum denominator
      
      # store ncc scores
      dmaps[:,:,displ] = s/np.sqrt(s_l*s_r)

      # pick best depth for each pixel
      best_map = np.argmax(dmaps,axis=2)+ start
    
    return best_map


def reconstruction_3d(disparity, X, Y, IL):


    # intrinsic parameter matrix
    fm = 403.657593 # Focal distantce in pixels
    cx = 161.644318 # Principal point - x-coordinate (pixels) 
    cy = 124.202080 # Principal point - y-coordinate (pixels) 
    bl = 119.929 # baseline (mm)
    # for the right camera    
    right_k = np.array([[ fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

    # for the left camera
    left_k = np.array([[fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

    disparity = np.array(np.reshape(depth_map,m*n))
    # Extrinsic parameters
    # Translation between cameras
    T = np.array([-bl, 0, 0]) 
    # Rotation
    R = np.array([[ 1,0,0],[ 0,1,0],[0,0,1]])

    pts_3d = []
    z_3d = []
    pixel_color = []

    for i in range(len(disparity)):

        z = (fm*bl)/disparity[i]

        if 1000 < z < 2000:

            pts = z*(np.dot(inv(left_k), np.array([X[i], Y[i], 1])).T)
            pts_3d.append(pts)

            pixel_color.append(IL[int(Y[i]),int(X[i])])


    pixel_color = np.asarray(pixel_color)
    pts_3d = np.array(pts_3d).T
    pts_3d = (pts_3d).tolist()
    pixel_color = np.asarray(pixel_color)


    return pts_3d , pixel_color

        
if __name__== "__main__":


    # Read images
    IL = cv2.imread('image/esquerda.ppm') # left image
    IR = cv2.imread('image/direita.ppm')  # right image
    gray1 = cv2.cvtColor(IL, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(IR, cv2.COLOR_BGR2GRAY)

    depth_map = plane_sweep_gauss(gray1,gray2,10,80,6)

    m,n,_ = IL.shape
    X,Y = np.meshgrid(np.arange(n),np.arange(m))
    X = np.reshape(X, m*n)
    Y = np.reshape(Y, m*n)

    disparity = np.array(np.reshape(depth_map,m*n)) # reshape do mesmo tamanho disparidade

    pts_3d, pixel_color = reconstruction_3d(disparity, X, Y, IL) 

    # Show images 
    fig, ax = plt.subplots(nrows=1, ncols=2)

    plt.subplot(1, 2, 1)
    plt.imshow(IL,cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(IR,cmap='gray')

    plt.figure()
    plt.imshow(depth_map, cmap = 'gray')

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(pts_3d[0], pts_3d[1], pts_3d[2], c = pixel_color/255)
    ax.view_init(elev=-80,azim=-90)

    plt.show()

