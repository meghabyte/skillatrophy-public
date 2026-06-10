
from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import cv2
import time

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import math


# ==============================================================================
# -- Calculate Error ----------------------------------------------------------
# ==============================================================================


num_collisions = 0
# Time of the most recent *hit event* (counted or not)
_last_hit_time = 0.0
_UNIQUENESS_WINDOW = 0.5  # seconds

def add_collision():
    """
    Called on every frame where a collision is detected.
    Count a collision only if there hasn't been any hit in the last 0.5s.
    """
    global num_collisions, _last_hit_time
    now = time.time()

    # Unique collision iff the previous hit was more than the window ago.
    if (now - _last_hit_time) > _UNIQUENESS_WINDOW:
        num_collisions += 1
        print("YOU HIT SOMETHING!")

    # Always update the last-hit timestamp on *every* hit frame
    _last_hit_time = now

def get_collision():
    """Returns the total number of unique collisions counted."""
    return num_collisions


def calc_distance(player, target: tuple):
    """
    Calculate the Euclidean distance between player and target

    Args:
        player: Carla world.player
        target: target distance (represented as a coordinate tuple) i.e. (10, 20)
    """
    player_location = player.get_location()
    #player_z = player_location.z 

    return math.sqrt((player_location.x - target[0])**2 + (player_location.y - target[1])**2)

def calc_tilt(player, target: int):
    """
    Calculate the error in tilt orientation between cars and agents

    Args:
        player: Carla world.player
        target: target tilt (heading) in degrees (int)
    """

    player_transform = player.get_transform()
    player_tilt = player_transform.rotation.yaw

    # NOTE: Carla heading is set to switch signs when you reach 180 degrees SE/SW. In other words:
    # if the target is positive, we must convert the angle to positive
    # if the target is negative, we must convert the angle to negative

    # not sure if this calculation is actually right, but we'll cross that bridge later
    # if target >= 0 and player_tilt < 0:
    #     player_tilt += 180
    # elif target < 0 and player_tilt > 0:
    #     player_tilt -= 180

    # I'm not sure if we want to differentiate between positive and negative here, so for now i'm just going to use absoluate value
    return min(abs(player_tilt - target), abs(player_tilt + target))

def calc_total_error(dist, tilt):
    """
    Calculate the total error of the run (this will decide whether the pass condition is met or not!)
    """
    error = dist + tilt * 0.2
    return error


def calc_score(dist, tilt, time):
    """
    Calculate the total score of the run, a weighted calculation including all variables

    # the score calculation is subject to change
    """

    score = dist + tilt * 3 + time
    return score

# ==============================================================================
# -- Collision Severity Metric ------------------------------------------------
# ==============================================================================

_total_collision_frames = 0

def increment_collision_frames():
    """Increments the counter for each frame a collision is active."""
    global _total_collision_frames
    _total_collision_frames += 1

def get_total_collision_frames():
    """Returns the total count of collision frames."""
    return _total_collision_frames

def reset_all_collisions():
    """Resets both unique and frame-based collision counters for a new trial."""
    global num_collisions, _total_collision_frames
    num_collisions = 0
    _total_collision_frames = 0