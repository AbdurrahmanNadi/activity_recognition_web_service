FROM  tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt -y update && apt -y upgrade && apt remove -y x264 libx264-dev \
    && apt install -y cmake \
    build-essential \
    checkinstall \
    yasm \
    git \
    pkg-config \
    libjpeg8-dev \
    libpng-dev \
    software-properties-common

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt -y update

RUN apt -y install libjasper1 \
    libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libdc1394-22-dev \
    libtiff-dev \
	libxine2-dev \
	libv4l-dev

RUN cd /usr/include/linux && ln -s -f ../libv4l1-videodev.h videodev.h && cd /root

RUN apt -y install libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	libgtk2.0-dev \
	libtbb-dev \
	qt5-default \
	libatlas-base-dev \
	libfaac-dev \
	libmp3lame-dev \
	libtheora-dev \
	libvorbis-dev \
	libxvidcore-dev \
	libopencore-amrnb-dev \
	libopencore-amrwb-dev \
	libavresample-dev \
	x264 \
	v4l-utils \
	libprotobuf-dev \
	protobuf-compiler \
	libgoogle-glog-dev \
	libgflags-dev \
	libgphoto2-dev \
	libeigen3-dev \
	libhdf5-dev \
	doxygen \
	python3-dev \
	python3-pip

RUN pip3 install -U pip numpy && apt install -y python3-testresources

RUN pip install wheel numpy scipy matplotlib

RUN mkdir /root/opencv_build && cd /root/opencv_build \
    && git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && mkdir -p opencv/build && cd opencv/build

RUN cd ~/opencv_build/opencv/build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
	-D WITH_QT=ON \
	-D WITH_OPENGL=ON \
	-D WITH_TBB=ON \
	-D WITH_V4L=ON ../

RUN cd ~/opencv_build/opencv/build && make -j8 && make install

RUN pip3 install "dm-sonnet<2" "tensorflow-probability<0.9.0" --pre

RUN pip3 install flask flask-RESTful

RUN mkdir -p /root/app

COPY . /root/app/

WORKDIR /root/app/

ENTRYPOINT python server.py
