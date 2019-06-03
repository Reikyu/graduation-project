Our graduation project theme is Wild Animal Catcher using Depth Camera.

Wild animals which lose their habitat and don't find feed show up at city and farm. So, produce and live stock are damaged and people receive a wound from wild animal's attack. Thus, there is a need to reduce wildlife that disturb the echosystem. To resolve this circumstance, robot that captures wildlife by using object detection tracking was devised. In the basic of python language, opencv and pyrealsense library was used. The equipments used are raspberrypi 3 B+, aduino r/c car, realsense D435.

There are some limitations to our project.
First, as you can see by looking at the code, color recognition is being used due to time and technical constraints. In order for this technology to be practical, it will be necessary to build a database of features extracted from photographs of wild animals and use object recognition technology. 

Second, since it is based on the assumption that one wild animal appears at a time, it will also need to be dealt with.
We measure the center coordinates and distance of an object to control its direction and velocity. if more than two target appears, there will be an error.

Third, we used it very low resolution because of the performance of the raspberry pie. f you use a high-performance PC, we recommend increasing resolution.
