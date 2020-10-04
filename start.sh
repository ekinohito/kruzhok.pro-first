#!/usr/bin/env bash

python3 -m venv .
source bin/activate
pip3 install opencv-python
pip3 install matplotlib
pip3 install numpy
chmod 777 ./main.py