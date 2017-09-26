# *****************************************************************************
# OpenDroneMap image / geodetic coordinates converter
# Copyright (c) 2017-2017, Henrik Dyrberg Egemose <hesc@mmmi.sdu.dk>
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

The class utilizes the utmconv class located in utm/utm.py.
The class uses the generated files in a OpenDroneMap project.
Functions check some inputs ranges but do not rely on it.
The class is implemented using python 3 and may not be backwards compatible. 

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

show_coord_on_images (image_and_points, folder)
    Use to visualize the found images and point from geodetic2images
    or orthophoto2images. Added as red circle around the point.
    image_and_points: dict of images as keys and u, v af values.
    folder: Place to save the images.

Revision
2017-09-26 HDE: Library created
"""
