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
testOdmConverter shows how the OdmConverter class can be used.

The script uses the OdmConverter class located in odmConverter.py

It is required that an ODM project have been made first. Since the software
uses the generated file from ODM

Revision
2017-09-26 HDE: Example script created
"""

from odmConverter import OdmConverter

# OpenDroneMap project folder
project_folder = ''  # The path where the ODM project in located '/home/ODM'
# init the class
ODM = OdmConverter(project_folder)
# if the use is to only convert image point to geodetic the initialization is
# faster with use of the only_image2geodetic=True i.e
# ODM = OdmConverter(project_folder, only_image2gps=True)


# use of image_point2geodetic
def image2geodetic():
    image = ''  # the name of a image in the ODM images folder, 'DJI_0001.JPG'
    uv = (1234, 321)  # the image coordinate measured in pixels from top left.
    ODM.set_image(image)
    lat, lon = ODM.image_point2geodetic(uv[0], uv[1])
    print((lat, lon))


# use of geodetic2images and show_coord_on_images
def geodetic2images():
    folder_out = ''  # A folder to save images in '/home/testGeo'
    latitude = 0.00  # latitude of geodetic point in degrees
    longitude = 0.00  # longitude of geodetic point in degrees
    image_and_points = ODM.geodetic2images(latitude, longitude)
    ODM.show_coord_on_images(image_and_points, folder_out)


# use of orthophoto2images and show_coord_on_images
def ortho2images():
    folder_out = ''  # A folder to save images in '/home/testOrtho'
    image_and_points = ODM.orthophoto2images(10252, 7514)
    ODM.show_coord_on_images(image_and_points, folder_out)


# use of image2orthophoto
def image2ortho():
    image = ''  # the name of a image in the ODM images folder, 'DJI_0001.JPG'
    ODM.set_image(image)
    x, y = ODM.image2orthophoto(3300, 1266)
    print((x, y))


if __name__ == '__main__':
    image2geodetic()
    geodetic2images()
    ortho2images()
    image2ortho()
