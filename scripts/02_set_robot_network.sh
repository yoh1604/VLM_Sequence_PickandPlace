#!/usr/bin/env bash
set -e

ROBOT_IF="enx9c69d31fbece"
PC_IP="192.168.200.12"
ROBOT_IP="192.168.200.1"

echo "Setting robot network on interface: $ROBOT_IF"

sudo ip addr flush dev enx9c69d31fbece
sudo ip addr add 192.168.200.12/24 dev enx9c69d31fbece
sudo ip link set enx9c69d31fbece up
sudo ip route replace 192.168.200.0/24 dev enx9c69d31fbece

echo "Checking route..."
ip route get 192.168.200.1

echo "Pinging robot..."
ping -c 4 192.168.200.1