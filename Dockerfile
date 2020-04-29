FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="abdurrahman.naddie@gmail.com"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTED noninteractive

WORKDIR /root/install

RUN apt update && apt install -y --no-install-recommends debconf-utils \
    apt-utils wget build-essential pkg-config git software-properties-common \
    gpg-agent cmake dirmngr

RUN apt install -y python3 python3-dev python3-pip
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    add-apt-repository ppa:jonathonf/ffmpeg-4 && \
    apt update

###################TensorRT Installation############################
#Nvidia Machine Learning Repo is included by default in the image
RUN apt update && apt install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.0 \
    libnvinfer-dev=6.0.1-1+cuda10.0 \
    libnvinfer-plugin6=6.0.1-1+cuda10.0
####################################################################
#
###################OpenCV Installation##############################
RUN apt install -y --no-install-recommends libjpeg8-dev libpng-dev libjasper1 \
    libavcodec-dev libavformat-dev libswscale-dev \
	libdc1394-22 libdc1394-22-dev libtiff-dev libxine2-dev \
	libv4l-dev libavresample-dev x264 \
	libx264-dev v4l-utils libprotobuf-dev \
	protobuf-compiler libgoogle-glog-dev libgflags-dev \
	libgphoto2-dev libeigen3-dev libhdf5-dev \
	doxygen libtbb-dev libatlas-base-dev \
	libfaac-dev libmp3lame-dev libtheora-dev \
	libvorbis-dev libxvidcore-dev libopencore-amrnb-dev \
	libopencore-amrwb-dev
RUN cd /usr/include/linux && ln -s -f ../libv4l1-videodev.h videodev.h

WORKDIR /root/install/opencv_build
RUN git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git

RUN pip3 install -U pip numpy && apt install -y python3-testresources
RUN pip install wheel scipy matplotlib

WORKDIR /root/install/opencv_build/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_opencv_python2=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=OFF \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D BUILD_DOCS=OFF ..
RUN make -j4 && make install

RUN ldconfig
#####################################################################

##########################Python Requirements########################
RUN pip install tensorflow-gpu==1.15
RUN pip install flask "dm-sonnet<2" "tensorflow-probability<0.9.0" --pre
#####################################################################

##########################CLEAN UP###################################
WORKDIR /root
RUN rm -rf install
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*
#####################################################################

WORKDIR /root/app
COPY ./ ./
ENTRYPOINT python3 server.py