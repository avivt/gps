""" This file defines utility functions for extracting fields out of mujoco xml model files. """
import re
import numpy as np


def extract_target(filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if re.search('name="target"', line):
                nums = re.findall('"([^"]*)"', line[line.find('pos'):])
                floats = [float(x) for x in nums[0].split()]
                return np.array(floats[0:2])
    return []


def extract_x0(filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if re.search('name="x0"', line):
                nums = re.findall('"([^"]*)"', line[line.find('pos'):])
                floats = [float(x) for x in nums[0].split()]
                return np.array([floats[0], floats[1], 0.0, 0.0])
    return []
