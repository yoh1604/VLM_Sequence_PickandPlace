#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/pick_place_occlusion_noetic"
source "$PROJECT_DIR/scripts/03_ros_env.sh"

ROBOT_IP="192.168.200.1"
KINEMATICS_CONFIG="$PROJECT_DIR/configs/ur5_calibration.yaml"

if [ -f "$KINEMATICS_CONFIG" ]; then
  echo "Launching UR5 driver with calibration: $KINEMATICS_CONFIG"
  roslaunch ur_robot_driver ur5_bringup.launch \
    robot_ip:="$ROBOT_IP" \
    kinematics_config:="$KINEMATICS_CONFIG"
else
  echo "Calibration file not found. Launching without kinematics_config."
  roslaunch ur_robot_driver ur5_bringup.launch \
    robot_ip:="$ROBOT_IP"
fi

roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.200.1 kinematics_config:="configs/ur5_calibration.yaml"