from odmConverter import OdmConverter


def image2gps():
    folder = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder += 'hojby-skra'
    odm = OdmConverter(folder)
    odm.set_image('DJI_0312.JPG')
    gps_pos = odm.image_point2gps(3300, 1266)
    print(gps_pos)


def gps2images():
    folder_in = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder_in += 'hojby-skra'
    folder_out = '/home/henrik/kode/droneMapAddon/testGps'
    latitude = 55.338876717598048
    longitude = 10.418817711878946
    odm = OdmConverter(folder_in)
    image_and_points = odm.gps2images(latitude, longitude)
    odm.show_coord_on_images(image_and_points, folder_out)


def ortho2images():
    folder_in = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder_in += 'hojby-skra'
    folder_out = '/home/henrik/kode/droneMapAddon/testOrtho'
    odm = OdmConverter(folder_in)
    image_and_points = odm.orthophoto2images(10252, 7514)
    odm.show_coord_on_images(image_and_points, folder_out)


if __name__ == '__main__':
    image2gps()
    gps2images()
    ortho2images()
