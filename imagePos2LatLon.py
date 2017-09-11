import cv2
import numpy as np
from pyquaternion import Quaternion
import re
from utm.utm import utmconv
from collections import namedtuple


class ImagePointToLatLon:
    def __init__(self, project):
        self.project = project
        self.reconstruction_file = project + '/opensfm/reconstruction.nvm'
        self.geo_model_file = project + \
            '/odm_georeferencing/odm_georeferencing_model_geo.txt'
        self.geo_transform_file = project + \
            '/odm_georeferencing/odm_georeferencing_transform.txt'

        self.geo_model_namedtuple = namedtuple('geo_model', ['hemisphere',
                                                             'zone',
                                                             'east_offset',
                                                             'north_offset'])
        self.reconstruction_namedtuple = namedtuple('reconstruction',
                                                    ['k', 'q', 'origin'])
        self.gps_namedtuple = namedtuple('gps', ['lat', 'lon', 'alt'])
        self.geo_model = self.get_geo_model()
        self.geo_transform = self.get_geo_transform()

    def get_geo_transform(self):
        geo_transform = np.zeros((4, 4))
        row_matcher = re.compile(' (-?\d+\.\d+),\t(-?\d+\.\d+),\t(-?\d+\.\d+),'
                                 '\t(-?\d+\.\d+) ')
        with open(self.geo_transform_file, 'r') as f:
            for i, line in enumerate(f):
                match = re.search(row_matcher, line)
                for j, num in enumerate(match.groups()):
                    geo_transform[i, j] = num
        return geo_transform

    def get_geo_model(self):
        with open(self.geo_model_file, 'r') as f:
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

    def get_reconstruction(self, image_name, image):
        k = None
        q = None
        origin = None
        row_matcher = re.compile('.*/' + image_name + ' (-?\d+\.\d+)'
                                 ' (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)'
                                 ' (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)'
                                 ' (-?\d+\.\d+) 0 0')
        with open(self.reconstruction_file, 'r') as f:
            for line in f:
                match = re.search(row_matcher, line)
                if match:
                    k = self.get_k(match.group(1), image)
                    q = Quaternion(match.group(2), match.group(3),
                                   match.group(4), match.group(5))
                    origin = np.array([float(match.group(6)),
                                       float(match.group(7)),
                                       float(match.group(8))])

        reconstruction = self.reconstruction_namedtuple(k, q, origin)
        return reconstruction

    @staticmethod
    def get_k(focal, image):
        k = np.zeros((3, 3))
        k[0, 0] = float(focal)
        k[1, 1] = float(focal)
        k[2, 2] = 1
        k[0, 2] = image.shape[1]/2
        k[1, 2] = image.shape[0]/2
        return k

    def get_depth(self, image_name, image, x, y):
        depth_map_file = self.project + '/opensfm/depthmaps/' + image_name \
                         + '.clean.npz'
        depth_map = np.load(depth_map_file)['depth']
        scale = depth_map.shape[1]/image.shape[1]
        x_new = round(scale * x)
        y_new = round(scale * y)
        depth = depth_map[y_new, x_new]
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
        hemisphere, zone, e_offset, n_offset = self.get_geo_model()
        east = p[0] + e_offset
        north = p[1] + n_offset
        utm_converter = utmconv()
        lat, lon = utm_converter.utm_to_geodetic(hemisphere, zone, east, north)
        pos = self.gps_namedtuple(lat, lon, p[2])
        return pos

    def get_lat_lon(self, image_name, x, y):
        image_file = self.project + '/images/' + image_name
        image = cv2.imread(image_file)
        depth = self.get_depth(image_name, image, x, y)
        recon = self.get_reconstruction(image_name, image)
        if recon.k is not None:
            point3d = self.get_3d_point(x, y, depth, recon)
            utm_point = self.get_geo_3d_point(point3d)
            pos = self.utm_to_lat_lon(utm_point)
        else:
            pos = False
        return pos


image_pos = ImagePointToLatLon('/home/henrik/droneMap/projects/hojby')
gps_pos = image_pos.get_lat_lon('DJI_0195.JPG', 1677, 1740)
print(gps_pos.lat)
print(gps_pos.lon)
