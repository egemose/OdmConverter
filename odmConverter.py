# *****************************************************************************
# OpenDroneMap converter between image and geodetic coordinates. It is part
# of InvaDrone, a research project by the University of Southern Denmark (SDU).
# Copyright (c) 2017, Henrik Dyrberg Egemose <hesc@mmmi.sdu.dk>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************
"""
OdmConverter implements conversion between geodetic coordinates and image
coordinates and between image and orthophoto.

The class utilizes the utmconv class from https://github.com/FroboLab/frobomind
The class uses the generated files from a OpenDroneMap project.
    http://opendronemap.org/ or https://github.com/OpenDroneMap/OpenDroneMap

Functions check some inputs ranges but do not rely on it.
The class is implemented using python 3 and may not be backwards compatible.

Dependencies:
    OpenCV
    python-magic (pip install python-magic)
    numpy (pip install numpy)
    pyquaternion (pip install pyquaternion)

OdmConverter uses the following generated files from OpenDroneMap projects:
    All images in the "images" folder.
    odm_georeferencing/odm_georeferencing_model_geo.txt
    odm_georeferencing/odm_georeferencing_transform.txt
    odm_orthophoto/odm_orthophoto.png
    odm_orthophoto/odm_orthophoto_corners.txt
    odm_texturing/odm_textured_model_geo.obj
    opensfm/reconstruction.nvm
    opensfm/camera_models.json
    All depthmaps from "opensfm/depthmaps" folder

OdmConverter (project, only_image_point2geodetic=False)
    Initialize the class for a given project (the folder for that project).
    If only_image_point2geodetic is True "image_point2geodetic" is the only
    function working, but initializing is faster.

set_image (image_name)
    Use to tell the class the current image to work on. Must be set before
    the first time image_point2geodetic is used.
    image_name is a string with the name of the image (not the path)

image_point2geodetic (u, v)
    Use to get the geodetic coordinate from image coordinates.
    u, v: The image coordinates measured from top left. [pixels]
    Returns: latitude [deg], longitude [deg]

geodetic2images (lat, lon)
    Use to get a list of images that can see the geodetic point
    lat: latitude [deg]
    lon: longitude [deg]
    Returns: dict of images as keys and u, v af values.

image2orthophoto (u, v)
    Use to get the corresponding point in the orthophoto
    u, v: The image coordinates measured from top left. [pixels]
    Returns: x, y Orthophoto coordinates measured from top left. [pixels]

orthophoto2images (u, v)
    Use to get a list of images that can see the orthophoto point.
    u, v: The orthophoto coordinates measured from top left. [pixels]
    Returns: dict of images as keys and u, v af values.

show_coord_on_images (image_and_points, folder, color=(0, 0, 255))
    Use to visualize the found images and point from geodetic2images
    or orthophoto2images. Added as red circle around the point.
    image_and_points: dict of images as keys and u, v af values.
    folder: Place to save the images.
    color: The color to draw with.

Revision
2017-09-26 HDE: Library created
"""
import glob
import json
import re
from collections import namedtuple
import cv2
import magic
import numpy as np
from pyquaternion import Quaternion
from utm.utm import utmconv


class ReconstructionError(Exception):
    """Error with the reconstruction"""
    pass


class NoCameraModelError(Exception):
    """Error with loading the camera model"""
    pass


class ImageSizeError(Exception):
    """Error with getting the image size"""
    pass


class NoImageError(Exception):
    """Error with getting the images"""
    pass


class MapError(Exception):
    """Error with the utm conversion"""
    pass


class Reconstruction:
    """Class to read and hold the reconstruction information from a
    OpenDroneMap project"""
    def __init__(self, project, only_image2geodetic):
        self.project = project
        self.regex_f = '(-?\d+\.\d+)'
        self.geo_model_namedtuple = namedtuple('geo_model', ['hemisphere',
                                                             'zone',
                                                             'east_offset',
                                                             'north_offset'])
        self.reconstruction_namedtuple = namedtuple('reconstruction',
                                                    ['k', 'q', 'origin'])
        self.geo_model = self.get_geo_model()
        self.geo_transform = self.get_geo_transform()
        self.image_list = self.list_of_images()
        self.model_image_shape = self.get_model_image_shape()
        self.image_name = None
        self.image_shape = None
        self.recon = None
        self.depth_map = None
        if not only_image2geodetic:
            self.model_3d = self.get_3d_model()
            self.ortho_corners = self.get_ortho_corners()
            self.ortho_size = self.get_ortho_size()

    def get_ortho_size(self):
        """read the size of of the ODM orthophoto"""
        image_file = self.project + '/odm_orthophoto/odm_orthophoto.png'
        regex = '(\d+) x (\d+)'
        size = self.get_image_info(image_file, regex)
        return size

    def get_ortho_corners(self):
        """read the utm coordinates for the ODM orthophoto corners"""
        file = self.project + '/odm_orthophoto/odm_orthophoto_corners.txt'
        regex = '(-?\d+\.\d+e\+\d+)'
        corner_matcher = re.compile((regex + ' ') * 3 + regex)
        with open(file, 'r') as f:
            for line in f:
                match = re.search(corner_matcher, line)
                if match:
                    p1 = np.array([float(match.group(1)),
                                   float(match.group(4))])
                    p2 = np.array([float(match.group(3)),
                                   float(match.group(2))])
                    corners = [p1, p2]
                    return corners

    def get_3d_model(self):
        """read the georeferenced 3d model"""
        model_file = self.project + '/odm_texturing/odm_textured_model_geo.obj'
        model_3d = []
        row_matcher = re.compile('v' + (' ' + self.regex_f) * 3)
        with open(model_file, 'r') as f:
            for line in f:
                match = re.search(row_matcher, line)
                if match:
                    point = np.array([float(match.group(1)),
                                      float(match.group(2)),
                                      float(match.group(3))])
                    model_3d.append(point)
        return model_3d

    def list_of_images(self):
        """Get at list of all images used for the ODM project"""
        image_folder = self.project + '/images/*.JPG'
        images = glob.glob(image_folder)
        return images

    def get_geo_transform(self):
        """Read the georeferencing transformation matrix"""
        geo_file = self.project \
                   + '/odm_georeferencing/odm_georeferencing_transform.txt'
        geo_transform = np.zeros((4, 4))
        row_matcher = re.compile((self.regex_f + ',\t') * 3 + self.regex_f)
        with open(geo_file, 'r') as f:
            for i, line in enumerate(f):
                match = re.search(row_matcher, line)
                for j, num in enumerate(match.groups()):
                    geo_transform[i, j] = num
        return geo_transform

    def get_geo_model(self):
        """Read the utm model used by ODM"""
        geo_file = self.project \
                   + '/odm_georeferencing/odm_georeferencing_model_geo.txt'
        with open(geo_file, 'r') as f:
            model = f.readline()
            offset = f.readline()
        hemisphere = model[-2]
        zone = int(model[-4:-2])
        east_offset_str, north_offset_str = offset.split(' ')
        east_offset = int(east_offset_str)
        north_offset = int(north_offset_str)
        geo_model = self.geo_model_namedtuple(hemisphere,
                                              zone,
                                              east_offset,
                                              north_offset)
        return geo_model

    def set_image(self, image_name):
        """Set the current image to be processed and load the
        corresponding reconstruction data"""
        image_file = self.project + '/images/' + image_name
        if image_file in self.image_list:
            self.image_name = image_name
            self.recon = self.get_reconstruction()
            self.depth_map = self.get_depth_map()
        else:
            raise NoImageError('Image (%s) is not part of the project.' %
                               image_name)

    def get_model_image_shape(self):
        """Get the size of the images used in the reconstruction. May be
        different from the input image size if ODM have resized the images"""
        shape = None
        file = self.project + '/opensfm/camera_models.json'
        with open(file, 'r') as f:
            camera_model = json.load(f)
            for cam in camera_model:
                cam_data = camera_model.get(cam)
                shape = (cam_data.get('width'), cam_data.get('height'))
        if shape is None:
            raise NoCameraModelError('No camera_models.json file found for '
                                     'the project.')
        return shape

    def get_image_size(self):
        """Get the size of the input images"""
        image_file = self.project + '/images/' + self.image_name
        regex = '(\d+)x(\d+)'
        size = self.get_image_info(image_file, regex)
        return size

    @staticmethod
    def get_image_info(image_file, regex):
        """Reads the image info and returns the size"""
        matcher = re.compile(regex)
        image_info = magic.from_file(image_file)
        match = re.search(matcher, image_info)
        if match:
            size = np.array([int(match.group(1)), int(match.group(2))])
        else:
            raise ImageSizeError('could not get the image size of image: '
                                 '(%s)' % image_file)
        return size

    def get_reconstruction(self):
        """Reads the reconstruction.nvm file and find the reconstruction
        corresponding to the current image"""
        recon = self.images_from_recon(self.image_name).__next__()
        if recon is None:
            raise ReconstructionError('Image \'%s\' is not part of the '
                                      'reconstruction.nvm file.' %
                                      self.image_name)
        return recon

    def images_from_recon(self, image_name=None):
        """Read the reconstruction.nvm file and return the reconstruction
        for each image"""
        if image_name is None:
            re_string = 'undistorted/(.*)'
        else:
            re_string = '(' + image_name + ')'
        reconstruction_file = self.project + '/opensfm/reconstruction.nvm'
        row_matcher = re.compile(re_string + (' ' + self.regex_f) * 8)
        with open(reconstruction_file, 'r') as f:
            for line in f:
                match = re.search(row_matcher, line)
                if match:
                    recon = self.parse_recon(match)
                    yield recon

    def parse_recon(self, match):
        """parses the reconstruction data into the corresponding variables"""
        self.image_name = match.group(1)
        self.image_shape = self.get_image_size()
        k = self.get_k(float(match.group(2)))
        q = Quaternion(float(match.group(3)), float(match.group(4)),
                       float(match.group(5)), float(match.group(6)))
        origin = np.array([float(match.group(7)),
                           float(match.group(8)),
                           float(match.group(9))])
        recon = self.reconstruction_namedtuple(k, q, origin)
        return recon

    def get_k(self, focal):
        """Computes the k matrix give the focal length"""
        d = np.append(focal * self.image_shape / self.model_image_shape, [1])
        k = np.diag(d)
        k[:2, 2] = self.image_shape / 2
        return k

    def get_depth_map(self):
        """Reads the depth map"""
        depth_map_file = self.project + '/opensfm/depthmaps/' \
                         + self.image_name + '.clean.npz'
        try:
            depth_map = np.load(depth_map_file)['depth']
        except FileNotFoundError:
            print('Image (%s) is not part of the reconstruction.'
                  % self.image_name)
            raise
        return depth_map

    def get_depth(self, u, v):
        """Get the depth at a given image coordinate, or closes non-zero
        depth"""
        scale = self.depth_map.shape[1] / self.image_shape[0]
        x_new = int(round(scale * u))
        y_new = int(round(scale * v))
        depth = self.depth_map[y_new, x_new]
        if depth == 0:
            depth = self.find_nearest_nonzero(self.depth_map, x_new, y_new)
            print('Warning: pixel (%i, %i) of image (%s) is not part of depth '
                  'map. \n'
                  'Nearest pixel in depth map is used, which may give some '
                  'offset in the position.' % (u, v, self.image_name))
        return depth

    @staticmethod
    def find_nearest_nonzero(depth_map, x, y):
        """Find the closes non-zero depth"""
        row, col = np.nonzero(depth_map)
        min_idx = ((row - y) ** 2 + (col - x) ** 2).argmin()
        depth = depth_map[row[min_idx], col[min_idx]]
        return depth


class OdmConverter:
    """Class for converting between image and geodetic coordinates and
    between image and orthophoto coordinates."""
    def __init__(self, project, *, only_image2geodetic=False):
        self.recon_model = Reconstruction(project, only_image2geodetic)
        self.utm = utmconv()

    def set_image(self, image_name):
        """Set the current image to be processed and load the corresponding
        data. Used before image_point2geodetic and image2orthophoto"""
        self.recon_model.set_image(image_name)

    def image2orthophoto(self, u, v):
        """With a point (u, v) in a image, finds the corresponding point in the
        orthophoto (both tiff and png)"""
        utm_point = self.image2utm(u, v)
        u_ortho, v_ortho = self.utm2orthophoto(utm_point[0], utm_point[1],
                                               self.recon_model.ortho_size,
                                               self.recon_model.ortho_corners)
        ortho = (int(u_ortho), int(v_ortho))
        return ortho

    def orthophoto2images(self, u, v):
        """With a given point (u, v) in the orthophoto (both tiff and png)
        finds the images where the point is visible together with the image
        coordinates. Returned as a dict with image names as key and
        coordinates as values"""
        geo_point = self.orthophoto2utm(u, v,
                                        self.recon_model.ortho_size,
                                        self.recon_model.ortho_corners)
        image_and_points = self.geo2images(geo_point)
        return image_and_points

    def show_coord_on_images(self, image_and_points, folder, color=(0, 0, 255)):
        """Reads the dict returns from orthophoto2images and geodetic2images
        and draw a circle around the point in the images and saves them in
        the given folder"""
        for image_name, point in image_and_points.items():
            image_file_in = self.recon_model.project + '/images/' + image_name
            image_file_out = folder + '/' + image_name
            image = cv2.imread(image_file_in)
            image = cv2.circle(image, point, 100, color, 10)
            cv2.imwrite(image_file_out, image)

    def geodetic2images(self, lat, lon):
        """Given geodetic coordinates it finds the images where the point
        is visible together with the image coordinates. Returned as a dict
        with image names as key and coordinates as values"""
        geo_point = self.geodetic2utm(lat, lon, self.recon_model.geo_model)
        image_and_points = self.geo2images(geo_point)
        return image_and_points

    def image_point2geodetic(self, u, v):
        """Given at image point (u, v) it finds the corresponding geodetic
        coordinates."""
        utm_point = self.image2utm(u, v)
        geodetic = self.utm2geodetic(utm_point, self.recon_model.geo_model)
        return geodetic

    @staticmethod
    def utm2orthophoto(x, y, ortho_size, ortho_corners):
        """Given a utm point it finds the corresponding orthophoto
        coordinates"""
        point = np.array([x, y])
        dp = ortho_corners[1] - ortho_corners[0]
        image_point = (point - ortho_corners[0]) * ortho_size / dp
        return image_point

    @staticmethod
    def orthophoto2utm(u, v, ortho_size, ortho_corners):
        """Given coordinates in the orthophoto finds the corresponding utm
        coordinates"""
        image_point = np.array([u, v])
        dp = ortho_corners[1] - ortho_corners[0]
        point = image_point * dp / ortho_size + ortho_corners[0]
        return point

    def geo2images(self, geo_point):
        """Given a utm coordinates finds the image where the point is visible
        together with the image coordinates. Return it as a dict with image
        name as key and coordinates as values."""
        image_and_points = {}
        geo_3d_point = self.utm2geo3d(geo_point, self.recon_model.model_3d)
        point3d = self.geo3d2point(geo_3d_point, self.recon_model.geo_transform)
        for recon in self.recon_model.images_from_recon():
            im_coord = self.world2image(point3d, recon)
            if self.check_output_coord(im_coord, self.recon_model.image_shape):
                image_and_points.update({self.recon_model.image_name: im_coord})
        return image_and_points

    def geodetic2utm(self, lat, lon, geo_model):
        """Convert geodetic coordinates to utm"""
        hemisphere, zone, e_offset, n_offset = geo_model
        hemisphere2, zone2, _, east, north = self.utm.geodetic_to_utm(lat, lon)
        if hemisphere == hemisphere2 and zone == zone2:
            geo_point = (east - e_offset, north - n_offset)
            return geo_point
        else:
            raise MapError('Conversion from geodetic to utm do not match the '
                           'geo model from OpenDroneMap. Check the geodetic '
                           'coordinates.')

    def image2utm(self, u, v):
        """Convert a image point (u, v) to utm coordinates"""
        self.check_input_coord(u, v, self.recon_model.image_shape)
        depth = self.recon_model.get_depth(u, v)
        point3d = self.image2world(u, v, depth, self.recon_model.recon)
        utm_point = self.point2utm(point3d, self.recon_model.geo_transform)
        return utm_point

    @staticmethod
    def utm2geo3d(geo_p, model_3d):
        """Convert utm to georeferenced 3d model point"""
        dist_keep = float('Inf')
        point_keep = 0
        geo_point = np.array(geo_p)
        for point in model_3d:
            dist = ((point[:2] - geo_point) ** 2).sum()
            if dist < dist_keep:
                dist_keep = dist
                point_keep = (geo_p[0], geo_p[1], point[2])
        return point_keep

    @staticmethod
    def geo3d2point(geo_p, geo_transform):
        """Converts georeferenced 3d model point to non georeferenced 3d model
        point"""
        geo_p_4d = np.array([geo_p[0], geo_p[1], geo_p[2], 1])
        p_4d = np.dot(np.linalg.inv(geo_transform), geo_p_4d)
        p = p_4d[0:3]
        return p

    @staticmethod
    def world2image(p, recon):
        """Converts 3d model point to image coordinates"""
        im_3d_point = np.dot(recon.k, recon.q.rotate(p - recon.origin))
        u, v = (im_3d_point[:2] / im_3d_point[2]).astype(int)
        return u, v

    @staticmethod
    def check_output_coord(coordinate, image_shape):
        """Checks if the given output image coordinates are within the image"""
        if 0 < coordinate[0] < image_shape[0]:
            if 0 < coordinate[1] < image_shape[1]:
                return True
        return False

    @staticmethod
    def check_input_coord(u, v, image_shape):
        """Raises a error if the input coordinates are outside the image"""
        if image_shape[0] < u:
            raise ValueError('x of %i is outside the image' % u)
        elif u < 0:
            raise ValueError('x must be positive, not %i' % u)
        if image_shape[1] < v:
            raise ValueError('y of %i is outside the image' % v)
        elif v < 0:
            raise ValueError('y must be positive, not %i' % v)

    @staticmethod
    def image2world(u, v, depth, recon):
        """Converts the image coordinates to 3d model point"""
        im_point = np.array([[u], [v], [1]])
        k_inv = np.linalg.inv(recon.k)
        qt = recon.q.inverse
        point = qt.rotate(depth * np.dot(k_inv, im_point)) + recon.origin
        return point

    @staticmethod
    def point2utm(p, geo_transform):
        """Convert a non georeferenced 3d model point to a georeferenced 3d
        model point"""
        p_4d = np.array([p[0], p[1], p[2], 1])
        geo_p_4d = np.dot(geo_transform, p_4d)
        geo_p = geo_p_4d[0:3]
        return geo_p

    def utm2geodetic(self, p, geo_model):
        """Converts a utm point to a geodetic point"""
        hemisphere, zone, e_offset, n_offset = geo_model
        east = p[0] + e_offset
        north = p[1] + n_offset
        lat, lon = self.utm.utm_to_geodetic(hemisphere, zone, east, north)
        return lat, lon
