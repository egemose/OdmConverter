import errno
import os
import re
from collections import namedtuple
import cv2
import numpy as np
from pyquaternion import Quaternion
from utm.utm import utmconv


class ReconstructionError(Exception):
    pass


class NoImageError(Exception):
    pass


class ImagePointToLatLon:
    def __init__(self, project):
        self.project = project
        self.regex_f = '(-?\d+\.\d+)'
        self.geo_model_namedtuple = namedtuple('geo_model', ['hemisphere',
                                                             'zone',
                                                             'east_offset',
                                                             'north_offset'])
        self.reconstruction_namedtuple = namedtuple('reconstruction',
                                                    ['k', 'q', 'origin'])
        self.gps_namedtuple = namedtuple('gps', ['lat', 'lon'])
        self.geo_model = self.get_geo_model()
        self.geo_transform = self.get_geo_transform()
        self.image = None
        self.image_name = None
        self.depth_map = None
        self.recon = None

    def get_geo_transform(self):
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

    def get_reconstruction(self):
        k = None
        q = None
        origin = None
        reconstruction_file = self.project + '/opensfm/reconstruction.nvm'
        row_matcher = re.compile(self.image_name + (' ' + self.regex_f) * 8)
        with open(reconstruction_file, 'r') as f:
            for line in f:
                match = re.search(row_matcher, line)
                if match:
                    k = self.get_k(float(match.group(1)), self.image)
                    q = Quaternion(float(match.group(2)), float(match.group(3)),
                                   float(match.group(4)), float(match.group(5)))
                    origin = np.array([float(match.group(6)),
                                       float(match.group(7)),
                                       float(match.group(8))])
        if k is None:
            raise ReconstructionError('Image \'%s\' is not part of the '
                                      'reconstruction.nvm file.' %
                                      self.image_name)
        reconstruction = self.reconstruction_namedtuple(k, q, origin)
        self.recon = reconstruction

    @staticmethod
    def get_k(focal, image):
        k = np.zeros((3, 3))
        k[0, 0] = focal
        k[1, 1] = focal
        k[2, 2] = 1
        k[0, 2] = image.shape[1] / 2
        k[1, 2] = image.shape[0] / 2
        return k

    def get_depth(self, image, x, y):
        scale = self.depth_map.shape[1] / image.shape[1]
        x_new = round(scale * x)
        y_new = round(scale * y)
        depth = self.depth_map[y_new, x_new]
        if depth == 0:
            depth = self.find_nearest_nonzero(self.depth_map, x_new, y_new)
            print('Warning: pixel (%i, %i) not part of depth model. '
                  'Nearest pixel in depth model is used, which may give some '
                  'offset in the position.' % (x, y))
        return depth

    def get_depth_map(self):
        depth_map_file = self.project + '/opensfm/depthmaps/' \
                         + self.image_name + '.clean.npz'
        try:
            depth_map = np.load(depth_map_file)['depth']
        except FileNotFoundError:
            print('Image is not part of the reconstruction')
            raise
        self.depth_map = depth_map

    @staticmethod
    def find_nearest_nonzero(depth_map, x, y):
        row, col = np.nonzero(depth_map)
        min_idx = ((row - y) ** 2 + (col - x) ** 2).argmin()
        depth = depth_map[row[min_idx], col[min_idx]]
        return depth

    @staticmethod
    def get_3d_point(x, y, depth, recon):
        im_point = np.array([[x], [y], [1]])
        k_inv = np.linalg.inv(recon.k)
        qt = recon.q.inverse
        point = qt.rotate(depth * np.dot(k_inv, im_point)) + recon.origin
        return point

    def get_geo_3d_point(self, p):
        p_4d = np.array([p[0], p[1], p[2], 1])
        geo_p_4d = np.dot(self.geo_transform, p_4d)
        geo_p = geo_p_4d[0:3]
        return geo_p

    def utm_to_lat_lon(self, p):
        hemisphere, zone, e_offset, n_offset = self.geo_model
        east = p[0] + e_offset
        north = p[1] + n_offset
        utm_converter = utmconv()
        lat, lon = utm_converter.utm_to_geodetic(hemisphere, zone, east, north)
        pos = self.gps_namedtuple(lat, lon)
        return pos

    def open_image(self, image_name):
        image_file = self.project + '/images/' + image_name
        image = cv2.imread(image_file)
        if image is None:
            print('Image not part of the project')
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    image_file)
        self.image = image
        self.image_name = image_name
        self.get_depth_map()
        self.get_reconstruction()

    @staticmethod
    def check_input_coordinates(image, x, y):
        if image.shape[1] < x:
            raise ValueError('x of %i is outside the image' % x)
        elif x < 0:
            raise ValueError('x must be positive, not %i' % x)
        if image.shape[0] < y:
            raise ValueError('y of %i is outside the image' % y)
        elif y < 0:
            raise ValueError('y must be positive, not %i' % y)

    def get_lat_lon(self, x, y):
        if self.image is None:
            raise NoImageError('Image has not been set.')
        self.check_input_coordinates(self.image, x, y)
        depth = self.get_depth(self.image, x, y)
        point3d = self.get_3d_point(x, y, depth, self.recon)
        utm_point = self.get_geo_3d_point(point3d)
        pos = self.utm_to_lat_lon(utm_point)
        return pos


if __name__ == '__main__':
    folder = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder += '2017-09-06-Pumpkins-50m-field1'
    image_pos = ImagePointToLatLon(folder)
    image_pos.open_image('DJI_0056.JPG')
    gps_pos = image_pos.get_lat_lon(4584, 1044)
    print(gps_pos.lat)
    print(gps_pos.lon)