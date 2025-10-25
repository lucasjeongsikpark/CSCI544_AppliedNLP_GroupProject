#!/bin/bash
# ==========================================
#  CARC FastAPI Port Forwarding Helper
#  Author: Jeongsik Park
#  Usage:
#     ./connect_carc_api.sh <node_name> <port> [username]
#  Example:
#     ./connect_carc_api.sh a04-20 8080
#     ./connect_carc_api.sh a04-20 8080 jeongsik
# ==========================================

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <node_name> <port> [username]"
    echo "Example: $0 a04-20 8080 jeongsik"
    exit 1
fi

NODE=$1
PORT=$2
USER=${3:-$USER}          # default: current system username
LOGIN_NODE="discovery.usc.edu"

echo "---------------------------------------------"
echo "🌐 CARC FastAPI Tunnel Setup"
echo "User:      $USER"
echo "Login:     $LOGIN_NODE"
echo "Compute:   $NODE"
echo "LocalPort: $PORT"
echo "---------------------------------------------"

# 1️⃣ SSH tunnel through login node to compute node
ssh -t -L ${PORT}:localhost:${PORT} ${USER}@${LOGIN_NODE} \
    "ssh -t -L ${PORT}:localhost:${PORT} ${NODE} \
     'echo \"✅ Connected to ${NODE}. You can now access FastAPI at http://localhost:${PORT}\"; bash'"
