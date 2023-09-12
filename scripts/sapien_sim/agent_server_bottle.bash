#!/bin/bash

echo "Initializing Agent Server"

configs_root_dir="configs/sapien_bottle"
server_name="agent"

python3 diffusion_edf/agent_server.py --configs-root-dir=$configs_root_dir \
                                      --server-name=$server_name \
                                    #   --init-nameserver
                                                