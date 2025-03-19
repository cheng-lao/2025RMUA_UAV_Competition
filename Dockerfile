FROM osrf/ros:noetic-desktop-full-focal

# 更换 Ubuntu 源为阿里云源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更换 ROS 源为中科大源
RUN sed -i 's/http:\/\/packages.ros.org/https:\/\/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/ros1-latest.list

ADD src /basic_dev/src/
ADD setup.bash /

RUN chmod +x /setup.bash

USER root

# 现在使用更换后的源进行更新和安装操作
RUN apt update && apt install -y python3-catkin-tools ros-noetic-geographic-msgs \
 ros-noetic-tf2-sensor-msgs ros-noetic-tf2-geometry-msgs ros-noetic-image-transport \
 net-tools

ENV ROS_DISTRO noetic

WORKDIR /basic_dev/
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make --only-pkg-with-deps airsim_ros && . devel/setup.sh && catkin_make --only-pkg-with-deps basic_dev

# ENTRYPOINT [ "/setup.bash" ]
CMD [ "bash", "launch_basic_dev.sh" ]