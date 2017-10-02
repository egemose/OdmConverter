# OdmConverter

OdmConverter uses [OpenDroneMap](https://github.com/OpenDroneMap/OpenDroneMap) generated files to convert between geodetic coordinates and image coordinates and between image and orthophoto coordinates.

## Getting Started

### Prerequisites

The content of a OpenDroneMap project. OpenDroneMap it self is not required.  
Python3  
OpenCV  
python-magic (pip install python-magic)  
numpy (pip install numpy)  
pyquaternion (pip install pyquaternion)  

### Installation

* open a terminal in the folder where OdmConverter shall be.
* Clone the repository:  `git clone https://github.com/OpenDroneMap/OpenDroneMap.git`

### Usage

Import the OdmConverter class in a python file:

```python
from odmConverter import OdmConverter
```

#### Class initializing and methods

##### Initialize the class
```python
OdmConverter(project, only_image_point2geodetic=False)
```
Initialize the class for a given ODM project.  
project: The path to the folder with the ODM project.  
If only_image_point2geodetic is True "image_point2geodetic" is the only function working, but initializing is faster.

##### Set current image

```python
set_image(image_name)
```
Use to tell the class the current image to work on.  
Must be set before the first time image_point2geodetic or image2orthophoto is used.  
image_name: is a string with the name of the image.

##### Convert a image point to geodetic

```python
image_point2geodetic(u, v)
```
Use to get the geodetic coordinate from image coordinates.  
u, v: The image coordinates measured from top left. [pixels]  
Returns: latitude [deg], longitude [deg]

##### Get a dictionary of images with the image coordinates of a geodetic point

```python
geodetic2images(lat, lon)
```
Use to get a list of images that can see the geodetic point  
lat: latitude [deg]  
lon: longitude [deg]  
Returns: dictionary of images as keys and u, v as values.  

##### Convert image point to a point on the orthophoto

```python
image2orthophoto(u, v)
```
Use to get the corresponding point in the orthophoto  
u, v: The image coordinates measured from top left. [pixels]  
Returns: x, y Orthophoto coordinates measured from top left. [pixels]

##### Get a dictionary of images with the image coordinates from a point on the orthophoto

```python
orthophoto2images(u, v)
```
Use to get a list of images that can see the orthophoto point.  
u, v: The orthophoto coordinates measured from top left. [pixels]  
Returns: dict of images as keys and u, v af values.  

##### Show the image coordinates recived from geodetic2images or orthophoto2images on the image

```python
show_coord_on_images(image_and_points, folder, color=(0, 0, 255))
```
Use to visualize the found images and point from geodetic2images or orthophoto2images. Added as red circle around the point.  
image_and_points: dict of images as keys and u, v af values.  
folder: Place to save the images.  
color: The color to draw with.

In [exampleOdmConverter.py](exampleOdmConverter.py) examples of how to use the OdmConverter class is showed.

## Author

Written by Henrik Dyrberg Egemose (hesc@mmmi.sdu.dk) as part of the InvaDrone project a research project by the University of Southern Denmark UAS Center (SDU UAS Center).

## License

This project is licensed under the 3-Clause BSD License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* The code uses files generated in a [OpenDroneMap](https://github.com/OpenDroneMap/OpenDroneMap) project.
* The utm/geodetic converter is made by Kjeld Jensen ([FroboMind](https://github.com/FroboLab/frobomind)).
