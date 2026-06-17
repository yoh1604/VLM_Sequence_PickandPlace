#!/usr/bin/env bash
set -e

sudo apt update

sudo apt install -y \
  git \
  curl \
  wget \
  build-essential \
  python3-pip \
  python3-venv \
  python3-catkin-tools \
  python3-rosdep \
  net-tools \
  iputils-ping \
  terminator \
  nano

sudo apt install -y \
  ros-noetic-moveit \
  ros-noetic-ur-robot-driver \
  ros-noetic-ur-description \
  ros-noetic-universal-robots

python3 -m pip install --upgrade pip

echo "Basic Ubuntu 20 + ROS Noetic dependencies installed."