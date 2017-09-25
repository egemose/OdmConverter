from odmConverter import OdmConverter


def gps2images():
    folder_in = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder_in += 'hojby'
    folder_out = '/home/henrik/kode/droneMapAddon/gpsConverter/test'
    latitude = 55.338876717598048
    longitude = 10.418817711878946
    odm = OdmConverter(folder_in)
    image_and_points = odm.gps2images(latitude, longitude)
    odm.show_coord_on_images(image_and_points, folder_out)


def image2gps():
    folder = '/home/henrik/mount/henrikServer/Documents/openDroneMap/projects/'
    folder += 'hojby'
    odm = OdmConverter(folder)
    odm.set_image('DJI_0275.JPG')
    gps_pos = odm.image_point2gps(543, 1002)
    print(gps_pos)


if __name__ == '__main__':
    image2gps()
    gps2images()
