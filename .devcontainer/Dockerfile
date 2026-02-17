FROM ubuntu:jammy

# Set the locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

# Set the timezone
ENV ROS_VERSION=2
ENV ROS_DISTRO=humble
ENV ROS_PYTHON_VERSION=3
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup the sources
RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y ros-humble-desktop 

# Install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    nano \
    iputils-ping \
    wget \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro humble

# Install pip and remove system scipy to prevent binary conflicts
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip && \
    apt-get remove -y python3-scipy && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

# Install Python dependencies with compatible versions
RUN pip install \
    numpy==2.2.6 \
    scipy>=1.13.0 \
    opencv-python \
    pynmea2 \
    # pyserial \
    pyproj \
    filterpy \
    rosbags \
    # rerun-sdk==0.23.0 \
    # 'rerun-sdk[notebook]' \
    debugpy \
    cython \
    matplotlib 
    # torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
    # ultralytics

ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && apt-get install -y \
    lsb-release gnupg \
    net-tools \
    vim \
    ffmpeg
    # rm -rf /var/lib/apt/lists/*


# Remove prebuilt cv_bridge to avoid ABI conflict with numpy 2.x
RUN apt-get remove -y ros-humble-cv-bridge || true
# Build cv_bridge from source for NumPy 2.2.6 compatibility
RUN apt-get update && apt-get install -y \
    python3-dev \
    libpython3-dev \
    python3-numpy \
    libopencv-dev

# WORKDIR /root/ros2_ws/src
# RUN git clone -b humble https://github.com/ros-perception/vision_opencv.git

WORKDIR /root/ros2_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select cv_bridge"
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN apt-get remove -y python3-sympy || true
RUN pip install ultralytics
RUN export PYTHONPATH=/root:${PYTHONPATH}
RUN apt install iproute2 -y
WORKDIR /root/ros2_ws

# Expose ports
EXPOSE 5902
EXPOSE 5678

CMD ["bash"]
