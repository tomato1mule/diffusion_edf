#!/bin/bash

echo "Initializing Agent Server"

configs_root_dir="configs/panda_bowl"
server_name="agent"
nameserver_host_ip="192.168.0.6"
nameserver_host_port="9090"

python3 diffusion_edf/agent_server.py --configs-root-dir=$configs_root_dir \
                                      --server-name=$server_name \
                                      --compile-score-model-head \
                                      --nameserver-host-ip=$nameserver_host_ip \
                                      --nameserver-host-port=$nameserver_host_port \
                                    #   --init-nameserver
                                                