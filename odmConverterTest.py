from odmConverter import OdmConverter

folder_in = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
folder_in += 'hojby-skra'
ODM = OdmConverter(folder_in)
# ODM = OdmConverter(folder_in, only_image2gps=True)


def image2gps():
    ODM.set_image('DJI_0312.JPG')
    gps_pos = ODM.image_point2gps(3300, 1266)
    print(gps_pos)


def gps2images():
    folder_out = '/home/henrik/kode/droneMapAddon/testGps'
    latitude = 55.338876717598048
    longitude = 10.418817711878946
    image_and_points = ODM.gps2images(latitude, longitude)
    ODM.show_coord_on_images(image_and_points, folder_out)


def ortho2images():
    folder_out = '/home/henrik/kode/droneMapAddon/testOrtho'
    image_and_points = ODM.orthophoto2images(10252, 7514)
    ODM.show_coord_on_images(image_and_points, folder_out)


if __name__ == '__main__':
    image2gps()
    gps2images()
    ortho2images()
