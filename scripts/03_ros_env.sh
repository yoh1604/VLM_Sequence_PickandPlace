#!/usr/bin/env bash

source /opt/ros/noetic/setup.bash
if [ -f "$HOME/ur5_noetic_ws/devel/setup.bash" ]; then
  source "$HOME/ur5_noetic_ws/devel/setup.bash"
elif [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
  source "$HOME/catkin_ws/devel/setup.bash"
fi

export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=192.168.200.12
unset ROS_HOSTNAME

echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
echo "ROS_HOSTNAME=$ROS_HOSTNAME"