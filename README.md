# MobileNet-SSDLite-RealSense-TF
RaspberryPi3(Raspbian Stretch) + MobileNetv2-SSDLite(Tensorflow/MobileNetv2SSDLite) + RealSense D435 + Tensorflow + without Neural Compute Stick(NCS)

## Change history
<details><summary>Change history</summary><div>
[Dec 02, 2018]　Corresponds to OpenCV3.4.3,Tensorflow v1.11.0, Protobuf 3.6.1, librealsense v2.16.5, D435 Firmware v5.10.6  
</div></details>  

## Environment
- RaspberryPi3 + Raspbian Stretch
- OpenCV 3.4.3 (Nov 25, 2018 updated)
- VFPV3 or TBB(Intel Threading Building Blocks)
- Tensorflow 1.11.0 (Nov 25, 2018 updated)
- Protobuf 3.6.1 (Nov 25, 2018 updated)
- librealsense v2.16.5 (Nov 25, 2018 updated)
- cmake 3.11.4
- MobileNetv2-SSDLite [MSCOCO]
- RealSense D435 (Firmware ver v5.10.6)
- Python3.5
- Numpy
- OpenGL

## Videos under test by Ubuntu 16.04

![Ubuntu1604](https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF/blob/master/media/MobileNet-SSDLite-TF-Ubuntu.gif)

## By RaspberryPi3 with OpenGL

![RaspberryPi3](https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF/blob/master/media/MobileNet-SSDLite-TF-Raspi-OpenGL.gif)


## RaspberryPi environment construction sequence
0.【Run in Ubuntu 16.04 environment】 Realsense D435's Firmware update
```bash
$ echo 'deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main' | sudo tee /etc/apt/sources.list.d/realsensepublic.list
$ sudo apt-key adv --keyserver keys.gnupg.net --recv-key 6F3EFCDE
$ sudo apt-get update
$ sudo apt-get install intel-realsense-dfu*
$ mkdir v5.10.6;cd v5.10.6
$ wget -O v5.10.6.zip https://downloadmirror.intel.com/28237/eng/Intel%C2%AE%20RealSense%E2%84%A2D400%20Series%20Signed%20Production%20Firmware%20v5_10_6.zip
$ unzip v5.10.6.zip

$ lsusb
### Below is sample.
Bus 002 Device 003: ID 8086:0b07 Intel Corp.
```
```bash
$ intel-realsense-dfu -b 002 -d 003 -f -i ./Signed_Image_UVC_5_10_6_0.bin
$ intel-realsense-dfu -p
FW version on device = 5.10.6.0
MM FW Version = 255.255.255.255
```
1.【Run in RaspberryPi environment】 Extend the SWAP area
```bash
$ sudo nano /etc/dphys-swapfile
CONF_SWAPSIZE=2048

$ sudo /etc/init.d/dphys-swapfile restart;swapon -s
```
2-1.Install tensorflow 1.11.0 (Raspbian Stretch)
```bash
$ sudo -H pip3 install pip --upgrade
$ sudo apt-get install python-pip python3-pip python3-scipy libhdf5-dev
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo pip3 uninstall tensorflow
$ wget -O tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc.whl
$ sudo pip3 install tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
【Required】Restart the terminal
```
2-2.Install tensorflow 1.11.0 (Ubuntu16.04 x86_64)
```bash
$ sudo -H pip3 install pip --upgrade
$ sudo -H pip3 install tensorflow==1.11.0 --upgrade
```

3.Install package and update udev rule
```bash
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
```bash
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
```bash
$ nano ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

$ source ~/.bashrc
```
6-1.Install protobuf 3.6.1 (Raspbian Stretch)
```bash
$ cd ~
$ wget https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
$ tar -zxvf protobuf-all-3.6.1.tar.gz
$ cd protobuf-3.6.1
$ ./configure
$ make -j1
$ sudo make install
$ nano ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/protobuf-3.6.1/src/.libs

$ source ~/.bashrc
$ cd python
$ python3 setup.py build --cpp_implementation 
$ python3 setup.py test --cpp_implementation
$ sudo python3 setup.py install --cpp_implementation
$ export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
$ export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
$ sudo ldconfig
$ protoc --version
```

6-2.Install protobuf 3.6.1 (Ubuntu16.04 x86_64)
```bash
$ cd git;mkdir protobuf;cd protobuf
$ wget -O protoc-3.6.1-linux-x86_64.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
$ unzip protoc-3.6.1-linux-x86_64.zip
$ rm protoc-3.6.1-linux-x86_64.zip
$ sudo mv -f bin/* /usr/local/bin/
$ sudo mv -bf include/* /usr/local/include/
$ sudo chown $USER /usr/local/bin/protoc
$ sudo chown -R $USER /usr/local/include/google
$ sudo ldconfig
$ nano ~/.bashrc
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2

$ source ~/.bashrc
```

7.Install TBB (Raspbian Stretch / Intel Threading Buiding Blocks)
```bash
$ cd ~
$ wget https://github.com/PINTO0309/TBBonARMv7/raw/master/libtbb-dev_2018U2_armhf.deb
$ sudo dpkg -i ~/libtbb-dev_2018U2_armhf.deb
$ sudo ldconfig
$ rm libtbb-dev_2018U2_armhf.deb
```
8.Install OpenCV 3.4.3(Raspbian Stretch / with TBB, with DNN, with OpenGL)
```bash
$ cd ~
$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.3.zip
$ unzip opencv.zip;rm opencv.zip
$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.3.zip
$ unzip opencv_contrib.zip;rm opencv_contrib.zip
$ cd ~/opencv-3.4.3/;mkdir build;cd build
$ cmake -D CMAKE_CXX_FLAGS="-DTBB_USE_GCC_BUILTINS=1 -D__TBB_64BIT_ATOMICS=0" \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.3/modules \
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
```bash
$ cd ~
$ git clone -b v2.16.5 https://github.com/IntelRealSense/librealsense.git
$ cd ~/librealsense
$ git checkout -b v2.16.5
$ mkdir build;cd build

$ cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release
OR
$ cmake .. -DBUILD_EXAMPLES=true

$ make -j1　# When running on a resource rich PC, "make -j8"
$ sudo make install
```
10.Install OpenCV Wrapper
```bash
$ cd ~/librealsense/wrappers/opencv;mkdir build;cd build
$ cmake ..
$ nano ../latency-tool/CMakeLists.txt
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
```bash
$ sudo apt-get install python-opengl
$ sudo -H pip3 install pyopengl
$ sudo -H pip3 install pyopengl_accelerate

$ sudo reboot
```
12-1.Introduction of model data of MobileNet-SSDLite (Raspbian Stretch)
```bash
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
12-2.Introduction of model data of MobileNet-SSDLite (Ubuntu16.04 x86_64)
```bash
$ mkdir tensorflow;cd tensorflow
$ git clone --recurse-submodules https://github.com/tensorflow/models.git
$ nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/home/<username>/tensorflow/models/research:/home/<username>/tensorflow/models/research/object_detection

$ source ~/.bashrc
$ cd ~/tensorflow/models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ cd ~/tensorflow/models/research/object_detection
$ wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```
13.Reduce the SWAP area to the default size
```bash
$ sudo nano /etc/dphys-swapfile
CONF_SWAPSIZE=100

$ sudo /etc/init.d/dphys-swapfile restart;swapon -s
```
14.Enable OpenGL Driver
```bash
$ sudo raspi-config
「7.Advanced Options」-「A7 GL Driver」-「G2 GL (Fake KMS)」 and Activate Raspberry Pi's OpenGL Driver
```
15.MobileNet-SSD execution
```bash
$ cd ~
$ git clone https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF.git
$ cd MobileNet-SSDLite-RealSense-TF
$ python3 MobileNetSSDwithRealSenseTF.py
```
