#!/bin/bash

# Inspired from https://github.com/takuseno/rl-ready-docker/blob/master/test.sh
python -c "import torch; import gymnasium as gym; env=gym.make('Hopper-v4')"
