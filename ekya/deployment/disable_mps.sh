#!/bin/bash
# a script to revert the environment back to shared GPU mode. Run this as sudo!
set -e

# Delete daemon
echo quit | nvidia-cuda-mps-control

## Disable exclusive mode
# Stop any display processes
service gdm stop

# Set default mode
nvidia-smi -i 0 -c DEFAULT

# Reboot GPU
nvidia-smi -i 0 -r
