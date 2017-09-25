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
    pass


class NoCameraModelError(Exception):
    pass


class ImageSizeError(Exception):
    pass


class NoImageError(Exception):
    pass


class Reconstruction:
    def __init__(self, project, *, only_image2gps):
        self.project = project
        self.regex_f = '(-?\d+\.\d+)'
        self.geo_model_namedtuple = namedtuple('geo_model', ['hemisphere',
                                                             'zone',
                                                             'east_offset',
                                                             'north_offset'])
        self.reconstruction_namedtuple = namedtuple('reconstruction',
                                                    ['k', 'q', 'origin'])
        self.point_namedtuple = namedtuple('point', ['x', 'y', 'z'])
        self.geo_model = self.get_geo_model()
        self.geo_transform = self.get_geo_transform()
        self.image_list = self.list_of_images()
        self.model_image_shape = self.get_model_image_shape()
        self.image_name = None
        self.image_shape = None
        self.recon = None
        self.depth_map = None
        if not only_image2gps:
            self.model_3d = self.get_3d_model()
            self.ortho_corners = self.get_ortho_corners()
            self.ortho_size = self.get_ortho_size()

    def get_ortho_size(self):
        image_file = self.project + '/odm_orthophoto/odm_orthophoto.png'
        image_info = magic.from_file(image_file)
        shape_matcher = re.compile('(\d+) x (\d+)')
        match = re.search(shape_matcher, image_info)
        if match:
            shape = (int(match.group(2)), int(match.group(1)))
        else:
            print('could not get the image size')
            raise ImageSizeError('could not get the image size of image: ('
                                 '%s)' % image_file)
        return shape

    def get_ortho_corners(self):
        file = self.project + '/odm_orthophoto/odm_orthophoto_corners.txt'
        corner_matcher = re.compile((self.regex_f + 'e\+(\d+) ') * 3 +
                                    self.regex_f + 'e\+(\d+)')
        with open(file, 'r') as f:
            for line in f:
                match = re.search(corner_matcher, line)
                if match:
                    x1 = float(match.group(1)) * 10 ** int(match.group(2))
                    y1 = float(match.group(3)) * 10 ** int(match.group(4))
                    x2 = float(match.group(5)) * 10 ** int(match.group(6))
                    y2 = float(match.group(7)) * 10 ** int(match.group(8))
                    corners = [(x1, y1), (x2, y2)]
                    return corners

    def get_3d_model(self):
        model_file = self.project + '/odm_texturing/odm_textured_model_geo.obj'
        model_3d = []
        row_matcher = re.compile('v' + (' ' + self.regex_f) * 3)
        with open(model_file, 'r') as f:
            for line in f:
                match = re.search(row_matcher, line)
                if match:
                    point = self.point_namedtuple(float(match.group(1)),
                                                  float(match.group(2)),
                                                  float(match.group(3)))
                    model_3d.append(point)
        return model_3d

    def list_of_images(self):
        image_folder = self.project + '/images/*.JPG'
        images = glob.glob(image_folder)
        return images

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

    def set_image(self, image_name):
        image_file = self.project + '/images/' + image_name
        if image_file in self.image_list:
            self.image_name = image_name
            self.update_recon_model()
        else:
            raise NoImageError('Image (%s) is not part of the project.' %
                               image_name)

    def update_recon_model(self):
        self.recon = self.get_reconstruction()
        self.depth_map = self.get_depth_map()

    def get_model_image_shape(self):
        shape = None
        file = self.project + '/opensfm/camera_models.json'
        with open(file, 'r') as f:
            camera_model = json.load(f)
            for cam in camera_model:
                cam_data = camera_model.get(cam)
                shape = (cam_data.get('height'), cam_data.get('width'))
        if shape is None:
            raise NoCameraModelError('No camera_models.json file found for '
                                     'the project.')
        return shape

    def get_image_shape(self):
        image_file = self.project + '/images/' + self.image_name
        image_info = magic.from_file(image_file)
        shape_matcher = re.compile('(\d+)x(\d+)')
        match = re.search(shape_matcher, image_info)
        if match:
            shape = (int(match.group(2)), int(match.group(1)))
        else:
            print('could not get the image size')
            raise ImageSizeError('could not get the image size of image: ('
                                 '%s)' % image_file)
        return shape

    def get_reconstruction(self):
        _, recon = self.images_from_recon(self.image_name).__next__()
        if recon is None:
            raise ReconstructionError('Image \'%s\' is not part of the '
                                      'reconstruction.nvm file.' %
                                      self.image_name)
        return recon

    def images_from_recon(self, image_name=None):
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
                    self.image_name = match.group(1)
                    self.image_shape = self.get_image_shape()
                    k = self.get_k(float(match.group(2)))
                    q = Quaternion(float(match.group(3)), float(match.group(4)),
                                   float(match.group(5)), float(match.group(6)))
                    origin = np.array([float(match.group(7)),
                                       float(match.group(8)),
                                       float(match.group(9))])
                    recon = self.reconstruction_namedtuple(k, q, origin)
                    yield self.image_name, recon

    def get_k(self, focal):
        k = np.zeros((3, 3))
        k[0, 0] = focal * self.image_shape[1] / self.model_image_shape[1]
        k[1, 1] = focal * self.image_shape[0] / self.model_image_shape[0]
        k[2, 2] = 1
        k[0, 2] = self.image_shape[1] / 2
        k[1, 2] = self.image_shape[0] / 2
        return k

    def get_depth_map(self):
        depth_map_file = self.project + '/opensfm/depthmaps/' \
                         + self.image_name + '.clean.npz'
        try:
            depth_map = np.load(depth_map_file)['depth']
        except FileNotFoundError:
            print('Image is not part of the reconstruction')
            raise
        return depth_map

    def get_depth(self, u, v):
        scale = self.depth_map.shape[1] / self.image_shape[1]
        x_new = round(scale * u)
        y_new = round(scale * v)
        depth = self.depth_map[y_new, x_new]
        if depth == 0:
            depth = self.find_nearest_nonzero(self.depth_map, x_new, y_new)
            print('Warning: pixel (%i, %i) not part of depth model. '
                  'Nearest pixel in depth model is used, which may give some '
                  'offset in the position.' % (u, v))
        return depth

    @staticmethod
    def find_nearest_nonzero(depth_map, x, y):
        row, col = np.nonzero(depth_map)
        min_idx = ((row - y) ** 2 + (col - x) ** 2).argmin()
        depth = depth_map[row[min_idx], col[min_idx]]
        return depth


class OdmConverter:
    def __init__(self, project, *, only_image2gps=False):
        self.recon_model = Reconstruction(project,
                                          only_image2gps=only_image2gps)
        self.utm = utmconv()

    def orthophoto2images(self, u, v):
        geo_point = self.orthophoto2utm(u, v,
                                        self.recon_model.ortho_size,
                                        self.recon_model.ortho_corners)
        image_and_points = self.geo2images(geo_point)
        return image_and_points

    @staticmethod
    def orthophoto2utm(u, v, ortho_size, ortho_corners):
        dx = ortho_corners[1][0] - ortho_corners[0][0]
        dy = ortho_corners[0][1] - ortho_corners[1][1]
        x = ortho_corners[0][0] + u/ortho_size[1] * dx
        y = ortho_corners[1][1] + v/ortho_size[0] * dy
        return x, y

    def show_coord_on_images(self, image_and_points, folder):
        for image_name, point in image_and_points.items():
            image_file_in = self.recon_model.project + '/images/' + image_name
            image_file_out = folder + '/' + image_name
            image = cv2.imread(image_file_in)
            image = cv2.circle(image, point, 100, (0, 0, 255), 10)
            cv2.imwrite(image_file_out, image)

    def gps2images(self, lat, lon):
        geo_point = self.gps2utm(lat, lon, self.recon_model.geo_model)
        image_and_points = self.geo2images(geo_point)
        return image_and_points

    def geo2images(self, geo_point):
        image_and_points = {}
        geo_3d_point = self.utm2geo3d(geo_point, self.recon_model.model_3d)
        point3d = self.geo3d2point(geo_3d_point, self.recon_model.geo_transform)
        for image_name, recon in self.recon_model.images_from_recon():
            im_coord = self.world2image(point3d, recon)
            if self.check_output_coord(im_coord, self.recon_model.image_shape):
                image_and_points.update({image_name: im_coord})
        return image_and_points

    def gps2utm(self, lat, lon, geo_model):
        hemisphere, zone, e_offset, n_offset = geo_model
        _, _, _, east, north = self.utm.geodetic_to_utm(lat, lon)
        geo_point = (east - e_offset, north - n_offset)
        return geo_point

    @staticmethod
    def utm2geo3d(geo_p, model_3d):
        dist_keep = 100000
        point_keep = 0
        for point in model_3d:
            dist = (point.x - geo_p[0]) ** 2 + (point.y - geo_p[1]) ** 2
            if dist < dist_keep:
                dist_keep = dist
                point_keep = (geo_p[0], geo_p[1], point.z)
        return point_keep

    @staticmethod
    def geo3d2point(geo_p, geo_transform):
        geo_p_4d = np.array([geo_p[0], geo_p[1], geo_p[2], 1])
        p_4d = np.dot(np.linalg.inv(geo_transform), geo_p_4d)
        p = p_4d[0:3]
        return p

    @staticmethod
    def world2image(p, recon):
        im_3d_point = np.dot(recon.k, recon.q.rotate(p - recon.origin))
        im_point = (int(round(im_3d_point[0] / im_3d_point[2])),
                    int(round(im_3d_point[1] / im_3d_point[2])))
        return im_point

    @staticmethod
    def check_output_coord(coordinate, image_shape):
        if 0 < coordinate[0] < image_shape[1]:
            if 0 < coordinate[1] < image_shape[0]:
                return True
        return False

    def set_image(self, image_name):
        self.recon_model.set_image(image_name)

    def image_point2gps(self, u, v):
        self.check_input_coord(u, v, self.recon_model.image_shape)
        depth = self.recon_model.get_depth(u, v)
        point3d = self.image2world(u, v, depth, self.recon_model.recon)
        utm_point = self.point2utm(point3d, self.recon_model.geo_transform)
        gps = self.utm2gps(utm_point, self.recon_model.geo_model)
        return gps

    @staticmethod
    def check_input_coord(u, v, image_shape):
        if image_shape[1] < u:
            raise ValueError('x of %i is outside the image' % u)
        elif u < 0:
            raise ValueError('x must be positive, not %i' % u)
        if image_shape[0] < v:
            raise ValueError('y of %i is outside the image' % v)
        elif v < 0:
            raise ValueError('y must be positive, not %i' % v)

    @staticmethod
    def image2world(u, v, depth, recon):
        im_point = np.array([[u], [v], [1]])
        k_inv = np.linalg.inv(recon.k)
        qt = recon.q.inverse
        point = qt.rotate(depth * np.dot(k_inv, im_point)) + recon.origin
        return point

    @staticmethod
    def point2utm(p, geo_transform):
        p_4d = np.array([p[0], p[1], p[2], 1])
        geo_p_4d = np.dot(geo_transform, p_4d)
        geo_p = geo_p_4d[0:3]
        return geo_p

    def utm2gps(self, p, geo_model):
        hemisphere, zone, e_offset, n_offset = geo_model
        east = p[0] + e_offset
        north = p[1] + n_offset
        lat, lon = self.utm.utm_to_geodetic(hemisphere, zone, east, north)
        return lat, lon
