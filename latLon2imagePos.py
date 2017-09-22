import errno
import glob
import json
import os
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


class LatLonToImagePoint:
    def __init__(self, project):
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
        self.model_3d = self.get_3d_model()
        self.image_list = self.list_of_images()
        self.image = None
        self.image_shape = None
        self.image_name = None
        self.depth_map = None
        self.recon = None

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
                    model_image_shape = self.get_model_image_shape()
                    k = self.get_k(float(match.group(1)),
                                   self.image_shape,
                                   model_image_shape)
                    q = Quaternion(float(match.group(2)), float(match.group(3)),
                                   float(match.group(4)), float(match.group(5)))
                    origin = np.array([float(match.group(6)),
                                       float(match.group(7)),
                                       float(match.group(8))])
        if k is None:
            return False
        reconstruction = self.reconstruction_namedtuple(k, q, origin)
        self.recon = reconstruction
        return True

    @staticmethod
    def get_k(focal, image_shape, model_image_shape):
        k = np.zeros((3, 3))
        k[0, 0] = focal * image_shape[1] / model_image_shape[1]
        k[1, 1] = focal * image_shape[0] / model_image_shape[0]
        k[2, 2] = 1
        k[0, 2] = image_shape[1] / 2
        k[1, 2] = image_shape[0] / 2
        return k

    @staticmethod
    def get_image_point(p, recon):
        im_3d_point = np.dot(recon.k, recon.q.rotate(p - recon.origin))
        im_point = (int(round(im_3d_point[0] / im_3d_point[2])),
                    int(round(im_3d_point[1] / im_3d_point[2])))
        return im_point

    def get_3d_point(self, geo_p):
        geo_p_4d = np.array([geo_p[0], geo_p[1], geo_p[2], 1])
        p_4d = np.dot(np.linalg.inv(self.geo_transform), geo_p_4d)
        p = p_4d[0:3]
        return p

    def lat_lon_to_utm(self, lat, lon):
        utm_converter = utmconv()
        hemisphere, zone, e_offset, n_offset = self.geo_model
        _, _, _, easting, northing = utm_converter.geodetic_to_utm(lat, lon)
        geo_3d_point = (round(easting - e_offset, 6),
                        round(northing - n_offset, 6))
        return geo_3d_point

    def open_image(self, image_name):
        image_file = self.project + '/images/' + image_name
        image = cv2.imread(image_file)
        if image is None:
            print('Image not part of the project')
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    image_file)
        self.image = image
        self.image_shape = image.shape
        self.image_name = image_name
        if self.get_reconstruction():
            return True
        return False

    def get_image_shape(self, image_name):
        image_file = self.project + '/images/' + image_name
        image_info = magic.from_file(image_file)
        shape_matcher = re.compile('(\d+)x(\d+)')
        match = re.search(shape_matcher, image_info)
        if match:
            shape = (int(match.group(2)), int(match.group(1)))
        else:
            print('could not get the image size')
            raise ImageSizeError('could not get the image size of image: ('
                                 '%s)' % image_file)
        self.image_shape = shape
        self.image_name = image_name
        if self.get_reconstruction():
            return True
        return False

    def check_coordinates(self, coordinate):
        if 0 < coordinate[0] < self.image_shape[1]:
            if 0 < coordinate[1] < self.image_shape[0]:
                return True
        return False

    def get_geo_3d_point(self, geo_p):
        dist_keep = 100000
        point_keep = 0
        for point in self.model_3d:
            dist = (point.x - geo_p[0]) ** 2 + (point.y - geo_p[1]) ** 2
            if dist < dist_keep:
                dist_keep = dist
                point_keep = (geo_p[0], geo_p[1], point.z)
        return point_keep

    def get_images_from_3d_point(self, point_3d):
        image_keep = None
        image_and_vertex = {}
        file = self.project + '/opensfm/reconstruction.meshed.json'
        with open(file, 'r') as f:
            shots = json.load(f)[0].get('shots')
            for shot, value in shots.items():
                image = shot
                dist_keep = 10000000
                for vertex in value.get('vertices'):
                    dist = (vertex[0] - point_3d[0]) ** 2 + \
                           (vertex[1] - point_3d[1]) ** 2 + \
                           (vertex[2] - point_3d[2]) ** 2
                    if dist < dist_keep:
                        dist_keep = dist
                        image_keep = image
                if dist_keep < 10:
                    image_and_vertex.update({image_keep: [point_3d,
                                                          dist_keep]})
        return image_and_vertex

    def get_images(self, lat, lon):
        images_and_point = {}
        geo_point = self.lat_lon_to_utm(lat, lon)
        geo_3d_point = self.get_geo_3d_point(geo_point)
        point3d = self.get_3d_point(geo_3d_point)
        images_and_vertex = self.get_images_from_3d_point(point3d)
        for image_name, vertex in images_and_vertex.items():
            if self.get_image_shape(image_name):
                pos = self.get_image_point(vertex[0], self.recon)
                if self.check_coordinates(pos):
                    images_and_point.update({image_name: [pos, vertex[1]]})
        return images_and_point

    def show_coordinate_on_images(self, lat, lon):
        images_and_point = {}
        geo_point = self.lat_lon_to_utm(lat, lon)
        geo_3d_point = self.get_geo_3d_point(geo_point)
        point3d = self.get_3d_point(geo_3d_point)
        images_and_vertex = self.get_images_from_3d_point(point3d)
        for image_name, vertex in images_and_vertex.items():
            if self.open_image(image_name):
                pos = self.get_image_point(vertex[0], self.recon)
                if self.check_coordinates(pos):
                    images_and_point.update({image_name: [pos, vertex[1]]})
                    temp = cv2.circle(self.image, pos, 50,
                                      (0, 0, 255), -1)
                    cv2.imwrite('/home/henrik/kode/droneMapAddon/test/' +
                                image_name, temp)
        return images_and_point


if __name__ == '__main__':
    folder = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder += 'hojby'
    latitude = 55.339205
    longitude = 10.419122
    gpsImage = LatLonToImagePoint(folder)
    images_and_points = gpsImage.get_images(latitude, longitude)
    for im, p_and_d in sorted(images_and_points.items(), key=lambda x: x[1][1]):
        print('image: ' + str(im))
        print('point: ' + str(p_and_d[0]))
        print('score: ' + str(p_and_d[1]))
        print(' ')
