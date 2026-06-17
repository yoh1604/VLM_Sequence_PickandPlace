#!/usr/bin/env bash

PROJECT_DIR="$HOME/pick_place_occlusion_noetic"
source "$PROJECT_DIR/scripts/03_ros_env.sh"

echo "=== ROS nodes ==="
rosnode list

echo ""
echo "=== Joint states sample ==="
rostopic echo /joint_states -n 1

echo ""
echo "=== Controllers ==="
rosservice call /controller_manager/list_controllers

echo ""
echo "=== FollowJointTrajectory topics ==="
rostopic list | grep follow_joint_trajectory || true

echo ""
echo "=== Speed topics ==="
rostopic list | grep speed || true