from util import *
import numpy as np
import cv2
import sys
import os

# Usage:
#     Change main function with ideal arguments
#     Then
#     from image_tranformer import ImageTransformer
#
# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image
#
# Reference:
#     1.        : http://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
#     2.        : http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html


class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, image_path, shape):
        self.image_path = image_path
        self.image = load_image(image_path, shape)

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_channels = self.image.shape[2]


    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)

        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):

        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])

        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])

        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))


# Usage:
#     Change main function with ideal arguments
#     then
#     python demo.py [name of the image] [degree to rotate] ([ideal width] [ideal height])
#     e.g.,
#     python demo.py images/000001.jpg 360
#     python demo.py images/000001.jpg 45 500 700
#
# Parameters:
#     img_path  : the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : the rotation around the x axis
#     phi       : the rotation around the y axis
#     gamma     : the rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image


# Input image path
img_path = "./Train/40/00040_00011_00026.png"

# Rotation range
rot_range = 320

# Ideal image shape (w, h)
img_shape = None

# Instantiate the class
it = ImageTransformer(img_path, img_shape)

# Make output dir
if not os.path.isdir('output'):
    os.mkdir('output')

# Iterate through rotation range
for ang in range(0, rot_range):

    # NOTE: Here we can change which angle, axis, shift

    """ Example of rotating an image along x-axis from 0 to 360 degree
        with a 5 pixel shift in +X direction """
    #rotated_img = it.rotate_along_axis(theta = ang, dx=5)

    """ Example of rotating an image along yz-axis from 0 to 360 degree """
    #rotated_img = it.rotate_along_axis(phi = ang, gamma = ang)

    """ Example of rotating an image along z-axis(Normal 2D) from 0 to 360 degree """
    #rotated_img = it.rotate_along_axis(gamma = ang)

    """ Example of rotating an image along x and y axis """
    rotated_img = it.rotate_along_axis(theta = ang, phi = ang)

save_image('output/{}.jpg'.format(str(ang).zfill(3)), rotated_img)
