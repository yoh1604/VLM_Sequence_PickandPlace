#!/usr/bin/env python3

import json
import sys
import rospy
import moveit_commander

WAYPOINT_FILE = "/home/b401/Documents/pick_place_occlusion_noetic/configs/waypoints_ur5.json"


def load_waypoint(name):
    with open(WAYPOINT_FILE, "r") as f:
        waypoints = json.load(f)

    if name not in waypoints:
        raise KeyError(f"Waypoint '{name}' tidak ditemukan di {WAYPOINT_FILE}")

    return waypoints[name]

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("go_to_idle_node", anonymous=True)

    group = moveit_commander.MoveGroupCommander("manipulator")

    group.set_max_velocity_scaling_factor(0.05)
    group.set_max_acceleration_scaling_factor(0.05)
    group.set_planning_time(5.0)
    group.set_num_planning_attempts(5)

    idle_joints = load_waypoint("IDLE")
    # idle_joints = load_waypoint("DISCARD_STEP_1") 

    rospy.loginfo("Current joints:")
    rospy.loginfo(group.get_current_joint_values())

    rospy.loginfo(f"Moving to IDLE: {idle_joints}")

    success = group.go(idle_joints, wait=True)

    group.stop()
    group.clear_pose_targets()

    if success:
        rospy.loginfo("Robot berhasil bergerak ke IDLE.")
    else:
        rospy.logerr("Gagal bergerak ke IDLE.")


if __name__ == "__main__":
    main()
