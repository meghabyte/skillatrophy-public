#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import time
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-s', '--start',
        metavar='S',
        default=0.0,
        type=float,
        help='starting time (default: 0.0)')
    argparser.add_argument(
        '-d', '--duration',
        metavar='D',
        default=0.0,
        type=float,
        help='duration (default: 0.0)')
    argparser.add_argument(
        '-f', '--recorder-filename',
        metavar='F',
        default="test_recording_01.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-c', '--camera',
        metavar='C',
        default=0,
        type=int,
        help='camera follows an actor (ex: 82)')
    argparser.add_argument(
        '-x', '--time-factor',
        metavar='X',
        default=1.0,
        type=float,
        help='time factor (default 1.0)')
    argparser.add_argument(
        '-i', '--ignore-hero',
        action='store_true',
        help='ignore hero vehicles')
    argparser.add_argument(
        '--move-spectator',
        action='store_true',
        help='move spectator camera')
    argparser.add_argument(
        '--spawn-sensors',
        action='store_true',
        help='spawn sensors in the replayed world')
    args = argparser.parse_args()

    try:

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # set to ignore the spectator camera or not
        client.set_replayer_ignore_spectator(args.move_spectator)

        world = client.get_world()

        # forcefully reset the world first
        # world.tick()
        # actors = list(world.get_actors().filter('vehicle.*'))
        # print(f'Attempting to delete {len(actors)} actors in custom_actor_spawn')
        # for actor in actors:
        #     try:
        #         actor.destroy()
        #     except Exception as e:
        #         print(f"Failed to destroy actor {actor}: {e}")
        
        spectator = world.get_spectator()
        current_rotation = spectator.get_transform().rotation
        new_transform = carla.Transform(carla.Location(-311, 155, 165), current_rotation)
        spectator.set_transform(new_transform)

        vehicles = world.get_actors().filter('vehicle.*')
        hero_vehicle = None
        print(vehicles)
        for vehicle in vehicles:
            is_hero = vehicle.attributes.get('role_name') == 'hero'
            #print(f"  - ID: {vehicle.id}, Type: {vehicle.type_id}, Is Hero: {is_hero}")
            if is_hero:
                hero_vehicle = vehicle
                break

        if hero_vehicle:
            transform = hero_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(x=-10, z=5), transform.rotation))
            hero_id = hero_vehicle.id
        else:
            print("No hero vehicle found.  Using default spectator location.")
            spectator = world.get_spectator()
            current_rotation = spectator.get_transform().rotation
            new_transform = carla.Transform(carla.Location(-311, 155, 165), current_rotation)
            spectator.set_transform(new_transform)
            hero_id = args.camera

        print(f'Hero ID: {hero_id}')

        if os.path.isabs(args.recorder_filename):
            replay_path = args.recorder_filename
        else:
            default_dir = '/home/driving_sim/CARLA_0.9.15/data'
            replay_path = os.path.join(default_dir, args.recorder_filename)

        print(client.replay_file(replay_path, args.start, args.duration, hero_id, args.spawn_sensors))

    finally:
        pass


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
