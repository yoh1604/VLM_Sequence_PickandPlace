# TERMINAL 1
source /opt/ros/noetic/setup.bash
source ~/ur5_noetic_ws/devel/setup.bash

cd ~/Documents/pick_place_occlusion_noetic

roslaunch ur_robot_driver ur5_bringup.launch \
  robot_ip:=192.168.200.1 \
  kinematics_config:=/home/b401/Documents/pick_place_occlusion_noetic/configs/ur5_calibration.yaml

# TERMINAL 2
source /opt/ros/noetic/setup.bash
source ~/ur5_noetic_ws/devel/setup.bash

roslaunch ur5_moveit_config moveit_planning_execution.launch

# TERMINAL 3
source /opt/ros/noetic/setup.bash
source ~/ur5_noetic_ws/devel/setup.bash

cd ~/Documents/pick_place_occlusion_noetic

roslaunch launch/gripper_tip_tf.launch

# TERMINAL 4
cd ~/Documents/pick_place_occlusion_noetic

./scripts/run_full_pick_pipeline.sh test_grasp_3 execute

python perception/capture_d455_once
python run_d455_full_pipeline
python models/graspnet-baseline/demo_d455.py \  --checkpoint_path models/graspnet-baseline/logs/log_rs/checkpoint.tar \  --num_point 20000 \  --num_view 300 \  --collision_thresh 0.01 \  --voxel_size 0.01 \  --no_vis
python perception/transform_grasp_to_base.py
python perception/convert_gripper_tip_to_tool0_target.py \
  --best_grasp_base outputs/$TEST_NAME/vision_output/best_grasp_base.json \
  --output outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json \
  --tool0_to_gripper_tip "0 0 0.17" \
  --pregrasp_z 0.10

export TEST_NAME=water_bottle

conda activate anygrasp_py310
python models/graspnet-baseline/demo_d455.py \
  --checkpoint_path models/graspnet-baseline/logs/log_rs/checkpoint.tar \
  --test_name $TEST_NAME \
  --num_point 20000 \
  --num_view 300 \
  --collision_thresh 0.01 \
  --voxel_size 0.01 \
  --no_vis

conda activate ur5_pickplace
python perception/transform_grasp_to_base.py

python perception/convert_gripper_tip_to_tool0_target.py \
  --best_grasp_base outputs/$TEST_NAME/vision_output/best_grasp_base.json \
  --output outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json \
  --tool0_to_gripper_tip "0 0 0.17" \
  --pregrasp_z 0.10

python3 robot_executor/nudge_tool0_target.py \
  --target_json outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json \
  --dx -0.05 \
  --dy 0.037

python3 robot_executor/robotiq_socket_control.py \
  --robot_ip 192.168.200.1 \
  open

python3 robot_executor/move_to_tool0_pregrasp.py \
  --target_json outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json

python3 robot_executor/execute_from_current_pregrasp_to_discard.py \
  --robot_ip 192.168.200.1 \
  --waypoints_json configs/waypoints_ur5.json \
  --grasp_down 0.05

# buat nudge
python3 robot_executor/nudge_tool0_target.py   --target_json outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json   --dx -0.05   --dy 0.035

# reset nudge
python perception/convert_gripper_tip_to_tool0_target.py \
  --best_grasp_base outputs/$TEST_NAME/vision_output/best_grasp_base.json \
  --output outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json \
  --tool0_to_gripper_tip "0 0 0.17" \
  --pregrasp_z 0.10

#   open grip
python3 robot_executor/robotiq_socket_control.py \
  --robot_ip 192.168.200.1 \
  open
