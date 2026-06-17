#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/pick_place_occlusion_noetic"

mkdir -p "$PROJECT_DIR"/{perception,planning,robot_executor,configs,data/d455_capture,outputs,logs/robot,logs/perception,logs/planning,logs/demo,scripts,models}

touch "$PROJECT_DIR"/outputs/.gitkeep
touch "$PROJECT_DIR"/logs/robot/.gitkeep
touch "$PROJECT_DIR"/logs/perception/.gitkeep
touch "$PROJECT_DIR"/logs/planning/.gitkeep
touch "$PROJECT_DIR"/logs/demo/.gitkeep

echo "Project structure created at: $PROJECT_DIR"