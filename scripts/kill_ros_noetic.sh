#!/usr/bin/env bash

echo "Killing ROS Noetic robot/moveit processes..."

pkill -9 -f rviz || true
pkill -9 -f move_group || true
pkill -9 -f roslaunch || true
pkill -9 -f rosmaster || true
pkill -9 -f roscore || true
pkill -9 -f ur_robot_driver || true
pkill -9 -f controller_manager || true
pkill -9 -f robot_state_publisher || true

sleep 2

echo "Remaining ROS processes:"
ps -ef | grep -E "roslaunch|rosmaster|roscore|move_group|rviz|ur_robot" | grep -v grep || true