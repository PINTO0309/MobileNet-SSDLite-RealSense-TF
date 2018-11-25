# MobileNet-SSDLite-RealSense-TF
RaspberryPi3(Raspbian Stretch) + MobileNet-SSDLite(Tensorflow/MobileNetSSDLite) + RealSense D435 + Tensorflow + without Neural Compute Stick(NCS)

## Environment
- RaspberryPi3 + Raspbian Stretch
- OpenCV 3.4.1
- VFPV3 or TBB(Intel Threading Building Blocks)
- Tensorflow 1.8.0
- Protobuf 3.5.1
- cmake 3.11.4
- MobileNet-SSDLite
- RealSense D435
- Python3.5
- Numpy
- OpenGL

## Videos under test by Ubuntu 16.04

![Ubuntu1604](https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF/blob/master/media/MobileNet-SSDLite-TF-Ubuntu.gif)

## By RaspberryPi3 with OpenGL

![RaspberryPi3](https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF/blob/master/media/MobileNet-SSDLite-TF-Raspi-OpenGL.gif)


## RaspberryPi environment construction sequence
1.Extend the SWAP area
```
$ sudo nano /etc/dphys-swapfile
CONF_SWAPSIZE=2048

$ sudo /etc/init.d/dphys-swapfile restart swapon -s
```
2.Install tensorflow 1.8.0
```
$ cd ~
$ sudo apt update;sudo apt upgrade
$ wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.8.0/tensorflow-1.8.0-cp35-none-linux_armv7l.whl
$ sudo pip3 install tensorflow-1.8.0-cp35-none-linux_armv7l.whl
$ rm tensorflow-1.8.0-cp35-none-linux_armv7l.whl
```
3.Install package and update udev rule
```
$ sudo pip3 install pillow lxml jupyter matplotlib cython
$ sudo apt install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev \
libglfw3-dev at-spi2-core libdrm* python-tk libjpeg-dev libtiff5-dev \
libjasper-dev libpng12-dev libavcodec-dev libavformat-dev \
libswscale-dev libv4l-dev libxvidcore-dev libx264-dev qt4-dev-tools \
autoconf automake libtool curl libatlas-base-dev
$ cd /etc/udev/rules.d/
$ sudo wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
$ sudo udevadm control --reload-rules && udevadm trigger
```
4.Install cmake-3.11.4
```
$ cd ~
$ wget https://cmake.org/files/v3.11/cmake-3.11.4.tar.gz
$ tar -zxvf cmake-3.11.4.tar.gz;rm cmake-3.11.4.tar.gz
$ cd cmake-3.11.4
$ ./configure --prefix=/home/pi/cmake-3.11.4
$ sudo make -j1
$ sudo make install
$ export PATH=/home/pi/cmake-3.11.4/bin:$PATH
$ source ~/.bashrc
$ cmake --version
cmake version 3.11.4
```
5.Update LD_LIBRARY_PATH
```
$ nano ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

$ source ~/.bashrc
```
6.Install protobuf 3.5.1
```
$ wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz
$ tar -zxvf protobuf-all-3.5.1.tar.gz
$ cd protobuf-3.5.1
$ ./configure
$ make -j1
$ sudo make install
$ cd python
$ export LD_LIBRARY_PATH=../src/.libs
$ python3 setup.py build --cpp_implementation 
$ python3 setup.py test --cpp_implementation
$ sudo python3 setup.py install --cpp_implementation
$ export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
$ export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
$ sudo ldconfig
$ protoc --version
```
7.Install TBB(Intel Threading Buiding Blocks)
```
$ cd ~
$ wget https://github.com/PINTO0309/TBBonARMv7/raw/master/libtbb-dev_2018U2_armhf.deb
$ sudo dpkg -i ~/libtbb-dev_2018U2_armhf.deb
$ sudo ldconfig
$ rm libtbb-dev_2018U2_armhf.deb
```
8.Install OpenCV 3.4.1(with TBB, with DNN, with OpenGL)
```
$ cd ~
$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.1.zip
$ unzip opencv.zip;rm opencv.zip
$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.1.zip
$ unzip opencv_contrib.zip;rm opencv_contrib.zip
$ cd ~/opencv-3.4.1/;mkdir build;cd build
$ cmake -D CMAKE_CXX_FLAGS="-DTBB_USE_GCC_BUILTINS=1 -D__TBB_64BIT_ATOMICS=0" \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.1/modules \
        -D BUILD_EXAMPLES=OFF \
        -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_opencv_python2=ON \
        -D BUILD_opencv_python3=ON \
        -D WITH_OPENCL=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=OFF \
        -D WITH_CUDA=OFF \
        -D ENABLE_NEON:BOOL=ON \
        -D ENABLE_VFPV3=ON \
        -D WITH_QT=OFF \
        -D BUILD_TESTS=OFF ..
$ make -j1
$ sudo make install
$ sudo ldconfig
```
9.Install Intel® RealSense™ SDK 2.0
```
$ cd ~
$ git clone https://github.com/IntelRealSense/librealsense.git
$ cd ~/librealsense;mkdir build;cd build

$ cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release
OR
$ cmake .. -DBUILD_EXAMPLES=true

$ make -j1
$ sudo make install
```
10.Install OpenCV Wrapper
```
$ cd ~/librealsense/wrappers/opencv;mkdir build;cd build
$ cmake ..
$ nano ../latency-tool/CMakeLists.txt

target_link_libraries(rs-latency-tool ${DEPENDENCIES})
↓
target_link_libraries(rs-latency-tool ${DEPENDENCIES} pthread)

$ make -j $(($(nproc) + 1))
$ sudo make install
$ cd ~/librealsense/build

#Python3.x
$ cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)

OR

#Python2.x
$ cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python)

$ make -j1
$ sudo make install
$ nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/usr/local/lib

$ source ~/.bashrc
```
11.Installing the OpenGL package for Python
```
$ sudo apt-get install python-opengl
$ sudo -H pip3 install pyopengl
$ sudo -H pip3 install pyopengl_accelerate

$ sudo reboot
```
12.Introduction of model data of MobileNet-SSDLite
```
$ mkdir tensorflow;cd tensorflow
$ git clone --recurse-submodules https://github.com/tensorflow/models.git
$ nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow/models/research/object_detection

$ source ~/.bashrc
$ cd ~/tensorflow/models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ cd ~/tensorflow/models/research/object_detection
$ wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```
13.Reduce the SWAP area to the default size
```
$ sudo nano /etc/dphys-swapfile
CONF_SWAPSIZE=100

$ sudo /etc/init.d/dphys-swapfile restart swapon -s
```
14.Enable OpenGL Driver
```
$ sudo raspi-config
「7.Advanced Options」-「A7 GL Driver」-「G2 GL (Fake KMS)」 and Activate Raspberry Pi's OpenGL Driver
```
15.MobileNet-SSD execution
```
$ cd ~
$ git clone https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF.git
$ cd MobileNet-SSDLite-RealSense-TF
$ python3 MobileNetSSDwithRealSenseTF.py
```
