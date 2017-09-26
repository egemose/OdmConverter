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
testOdmConverter shows how the OdmConverter can be used.

The script uses the OdmConverter class located in odmConverter.py
The script is implemented using python 3 and may not be backwards compatible.

Revision
2017-09-26 HDE: Example script created
"""

from odmConverter import OdmConverter

# OpenDroneMap projects folder
folder_in = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
# The project of interest
folder_in += 'hojby-skra'
# init the class
ODM = OdmConverter(folder_in)
# ODM = OdmConverter(folder_in, only_image2gps=True)


# use of image_point2geodetic
def image2geodetic():
    ODM.set_image('DJI_0312.JPG')
    geodetic = ODM.image_point2geodetic(3300, 1266)
    print(geodetic)


# use of geodetic2images and show_coord_on_images
def geodetic2images():
    folder_out = '/home/henrik/kode/droneMapAddon/testGps'
    latitude = 55.338876717598048
    longitude = 10.418817711878946
    image_and_points = ODM.geodetic2images(latitude, longitude)
    ODM.show_coord_on_images(image_and_points, folder_out)


# use of orthophoto2images and show_coord_on_images
def ortho2images():
    folder_out = '/home/henrik/kode/droneMapAddon/testOrtho'
    image_and_points = ODM.orthophoto2images(10252, 7514)
    ODM.show_coord_on_images(image_and_points, folder_out)


# use of image2orthophoto
def image2ortho():
    ODM.set_image('DJI_0312.JPG')
    x, y = ODM.image2orthophoto(3300, 1266)
    print((x, y))


if __name__ == '__main__':
    image2geodetic()
    geodetic2images()
    ortho2images()
    image2ortho()
