# Created by Hailiang Tang

FROM    nvidia/cudagl:11.4.2-devel-ubuntu20.04


# TODO: For Mainland China
COPY    sources.list/sources.list /etc/apt/
COPY    sources.list/ros-latest.list /etc/apt/sources.list.d/

# Update software mirror and update zone
RUN     mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.save && \
        apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F42ED6FBAB17C654 15CF4D18AF4F7421 && \
        echo 'Etc/UTC' > /etc/timezone && \
        ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN     apt-get update && \
        apt-get upgrade -q -y && \
        apt-get install -q -y tzdata && \
        rm -rf /var/lib/apt/lists/*

# Common
RUN     apt-get update && \
        apt-get install -y cmake gcc g++ gdb build-essential git && \
        rm -rf /var/lib/apt/lists/*

# ROS
RUN     apt-get update && \
        apt-get install -y --no-install-recommends python3-rosdep python3-rosinstall ros-noetic-ros-core \
        ros-noetic-ros-base ros-noetic-catkin ros-noetic-perception python3-catkin-tools && \
        rm -rf /var/lib/apt/lists/*
# TODO: For Mainland China
RUN     apt-get update && \
        apt-get install -y python3-pip && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install rosdepc && \
        rosdepc init && rosdepc update --rosdistro noetic
# RUN     rosdep init && rosdep update --rosdistro noetic

# Livox
COPY    libraries/Livox-SDK-2.3.0.tar.gz /root
RUN     cd /root && \
        tar -xf Livox-SDK-2.3.0.tar.gz && \
        cd Livox-SDK-2.3.0/build && \
        cmake -DCMAKE_BUILD_TYPE=Release .. && \
        make install -j15 && \
        rm -rf /root/Livox-SDK-2.3.0

# abseil-cpp
COPY    libraries/abseil-cpp-20260107.1.tar.gz /root
RUN     cd /root && \
        tar -xf abseil-cpp-20260107.1.tar.gz && \
        cd abseil-cpp-20260107.1 && \
        mkdir build && cd build && \
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_CXX_STANDARD=17 .. && \
        make install -j15 && \
        rm -rf /root/abseil-cpp-20260107.1 /root/abseil-cpp-20260107.1.tar.gz

# oneTBB
COPY    libraries/oneTBB-2023.1.0.tar.gz /root
RUN     cd /root && \
        tar -xf oneTBB-2023.1.0.tar.gz && \
        cd oneTBB-2023.1.0 && \
        mkdir build && cd build &&\
        cmake -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=false .. && \
        make install -j15 && \
        rm -rf /root/oneTBB-2023.1.0

# ceres-solver
COPY    libraries/ceres-solver-2.2.0.tar.gz /root
RUN     apt-get update && \
        apt-get install -y libatlas-base-dev libeigen3-dev libsuitesparse-dev libgoogle-glog-dev && \
        rm -rf /var/lib/apt/lists/* && \
        cd /root && \
        tar -xf ceres-solver-2.2.0.tar.gz && \
        cd ceres-solver-2.2.0 && \
        mkdir build && cd build && \
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=true -DBUILD_TESTING=false -DBUILD_EXAMPLES=false .. && \
        make install -j15 && \
        rm -rf /root/ceres-solver-2.2.0

# OpenCV
COPY    libraries/opencv-4.12.0.tar.gz /root
COPY    libraries/opencv_contrib-4.12.0.tar.gz /root
RUN     cd /root && \
        tar -xf opencv-4.12.0.tar.gz && \
        tar -xf opencv_contrib-4.12.0.tar.gz && \
        cd opencv-4.12.0 && \
        mkdir build && cd build && \
        cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-4.12.0/modules \
        -DWITH_CUDA=true -DWITH_TBB=true -DBUILD_opencv_python2=false -DBUILD_opencv_python3=false -DBUILD_JAVA=false \
        -DBUILD_opencv_dnn=false -DOPENCV_DNN_CUDA=false -DBUILD_PERF_TESTS=false -DBUILD_TESTS=false \
        -DBUILD_EXAMPLES=false -DBUILD_opencv_sfm=false -DBUILD_opencv_xfeatures2d=false -DBUILD_opencv_xphoto=false \
        -DBUILD_opencv_objdetect=false -DBUILD_opencv_cudaobjdetect=false -DBUILD_opencv_stitching=false  \
        -DWITH_QT=true -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF .. && \
        make install -j15 && \
        rm -rf /root/opencv-4.12.0 /root/opencv_contrib-4.12.0

# New user
ENV     USER_NAME=lvio
ENV     HOME=/home/$USER_NAME

USER    root
RUN     apt-get update && \
        apt-get install -y zsh curl && \
        rm -rf /var/lib/apt/lists/* && \
        chsh -s /usr/bin/zsh && \
        useradd -ms /usr/bin/zsh $USER_NAME

# Change user to non-root
USER    $USER_NAME
WORKDIR $HOME

# Livox ROS Driver
RUN     cd $HOME && \
        mkdir catkin_ws && \
        mkdir catkin_ws/src && \
        cd catkin_ws/src && \
        git clone https://github.com/livox-SDK/livox_ros_driver && \
        . /opt/ros/noetic/setup.sh && \
        cd $HOME/catkin_ws && \
        catkin_make -DCMAKE_BUILD_TYPE=Release

# Workspace
RUN     sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
        echo "source /opt/ros/noetic/setup.zsh" >> $HOME/.zshrc && \
        echo "source $HOME/catkin_ws/devel/setup.zsh" >> $HOME/.zshrc && \
        echo "source $HOME/workspace/devel/setup.zsh" >> $HOME/.zshrc && \
        mkdir $HOME/workspace && \
        mkdir $HOME/workspace/src