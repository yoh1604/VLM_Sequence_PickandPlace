#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/pick_place_occlusion_noetic"
source "$PROJECT_DIR/scripts/03_ros_env.sh"

roslaunch ur5_moveit_config moveit_planning_execution.launch