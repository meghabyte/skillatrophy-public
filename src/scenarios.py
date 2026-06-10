import carla

# ==============================================================================
# -- SCENARIOS ---------------------------------------------------------------
# ==============================================================================

def custom_actor_spawn(client, position: str, dist_dev=0, barrier_d=2):
    # Get the world
    world = client.get_world()

    vehicle = None
    vehicle_list = []
    try:
        # TEMPORARILY DISABLED
        # if position == "Parallel Parking":
        #     vehicle_list = parallel_parking(client.get_world(), dist_dev, barrier_d)
        # elif position == "Angled Parking (Incline)":
        #     vehicle_list = angled_incline(client.get_world(), dist_dev)
        # elif position == "Angled Parking (Flat)":
        #     vehicle_list = angled_flat(client.get_world(), dist_dev)
        # elif position == "Standard Parking":
        #     vehicle_list = standard_parking(client.get_world(), dist_dev)
        # else:
        #     raise Exception("Invalid position parameter in custom_actor_spawn")

        vehicle_list.extend(angled_flat(client.get_world())) # angled_flat spawn
        print("Successfully spawned angled_flat scenarios")
        vehicle_list.extend(standard_parking_combined(client.get_world())) # standard parking spawn
        print("Successfully spawned standard_parking scenarios")
        vehicle_list.extend(parallel_parking_combined(client.get_world())) # parallel parking spawn
        print("Successfully spawned parallel_parking scenarios")
        vehicle_list.extend(parallel_parking_combined_slippery(client.get_world())) # parallel parking (slippery) spawn
        print("Successfully spawned parallel_parking_slippery scenarios")


        print("Vehicle Actor spawned successfully.")
    except Exception as e:
        raise Exception(f'Encountered the following error in custom_actor_spawn when attempting to create vehicle(s):\n{e}')

    for _ in range(50):
        client.get_world().tick()

    # Disable vehicle physics (to keep it stationary)
    for vehicle in vehicle_list:
        vehicle.set_simulate_physics(False)

def parallel_parking_combined(world):
    # list of vehicle spawnpoints and orientations, in format:
    # [x, y, z, heading]
    vehicle_spawn_points = [
        [-341, 141.5, 161, 180],
        [-346.4, 141.5, 161, 180],
        [-351.4, 141.5, 161, 180], # Cover 15m gap (Added 1/9/2025)
        [-356.8, 141.5, 161, 180], # 15m gap
        [-369.8, 141.5, 161, 180], # 13m gap
        [-375.0, 141.5, 161, 180], # Cover 11m gap (Added 1/9/2025)
        [-380.8, 141.5, 161, 180], # 11m gap
    ]

    # vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # return list
    spawned_vehicles = []

    # spawn each vehicles from vehicle_spawn_points
    for x_start, y_start, z_start, heading in vehicle_spawn_points:
        spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
        spawned_vehicles.append(world.spawn_actor(vehicle_blueprint, spawn_transform))

    return spawned_vehicles

def parallel_parking_combined_slippery(world):
    # list of vehicle spawnpoints and orientations, in format:
    # [x, y, z, heading]
    vehicle_spawn_points = [
        [-271.8, 141.5, 161, 180],
        [-260.8, 141.5, 161, 180], # 11m gap
        [-247.8, 141.5, 161, 180], # 13m gap
        [-232.8, 141.5, 161, 180], # 15m gap
    ]

    # vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # return list
    spawned_vehicles = []

    # spawn each vehicles from vehicle_spawn_points
    for x_start, y_start, z_start, heading in vehicle_spawn_points:
        spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
        spawned_vehicles.append(world.spawn_actor(vehicle_blueprint, spawn_transform))

    return spawned_vehicles


def parallel_parking(world, distance, barrier_d=0):
    return []

    # Get a vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # NOTE: in this parallel parking scenario, only the agent's x axis has to change.s

    # Vehicle 1
    x_start, y_start, z_start, heading = -313, 160, 162, -3
    spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
    vehicle1 = world.spawn_actor(vehicle_blueprint, spawn_transform)
    print('vehicle 1 successfully spawned')
    
    # Vehicle 2
    distance = max(distance, 0) # ensure that distance is ALWAYS positive
    x_start, y_start, z_start, heading = -(323 + distance), 160, 162, -3
    spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
    vehicle2 = world.spawn_actor(vehicle_blueprint, spawn_transform)
    print('vehicle 2 successfully spawned')

    vehicle1.apply_control(carla.VehicleControl(brake=1.0))
    vehicle2.apply_control(carla.VehicleControl(brake=1.0))

    # Spawn barrier (frozen vehicles)
    """
    for d in range(6):
        x_start, y_start, z_start, heading = -(307 + 5.7 * d), 155 - barrier_d, 162, -3
        spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
        barrier_vehicle = world.spawn_actor(vehicle_blueprint, spawn_transform)
        barrier_vehicle.apply_control(carla.VehicleControl(brake=1.0))
    """
    

    return [vehicle1, vehicle2]

def angled_incline(world): #THIS NEEDS TO BE FIXED
    # Get a vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # Define the spawn location and rotation
    x_start, y_start, z_start, heading = -565.5, 178.8, 167, 160
    spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_transform)

    return [vehicle]

def angled_flat(world):
    # list of vehicle spawnpoints and orientations, in format:
    # [x, y, z, heading]
    vehicle_spawn_points = [
        [-294.5, 158.7, 160, 69],
        [-301.5, 158.7, 160, 69], # 7m gap
        [-307.5, 158.7, 160, 69], # 6m gap
        [-312.5, 158.7, 160, 69], # 5m gap
        [-317.0, 158.7, 160, 69], # 4.5m gap
    ]

    # vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # return list
    spawned_vehicles = []

    # spawn each vehicles from vehicle_spawn_points
    for x_start, y_start, z_start, heading in vehicle_spawn_points:
        spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
        spawned_vehicles.append(world.spawn_actor(vehicle_blueprint, spawn_transform))

    return spawned_vehicles

def standard_parking_combined(world):
    # list of vehicle spawnpoints and orientations, in format:
    # [x, y, z, heading]
    vehicle_spawn_points = [
        [-320, 158.7, 160, 88],
        [-324.5, 158.7, 160, 88], # 4.5m gap
        [-329.5, 158.7, 160, 88], # 5m gap
        [-335.5, 158.7, 160, 88], # 6m gap
        #[-342.5, 158.7, 160, 88], # 7m gap X* not enough room in environment to spawn this
    ]

    # vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # return list
    spawned_vehicles = []

    # spawn each vehicles from vehicle_spawn_points
    for x_start, y_start, z_start, heading in vehicle_spawn_points:
        spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
        spawned_vehicles.append(world.spawn_actor(vehicle_blueprint, spawn_transform))

    return spawned_vehicles

def standard_parking(world, distance):

    # Get a vehicle blueprint
    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0] # set to tesla model 3

    # NOTE: in this parallel parking scenario, only the agent's x axis has to change.s

    # Vehicle 1
    x_start, y_start, z_start, heading = -306.9, 158.9, 162, 89
    spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
    vehicle1 = world.spawn_actor(vehicle_blueprint, spawn_transform)
    print('vehicle 1 successfully spawned')
    
    # Vehicle 2
    distance = max(distance, 0) # ensure that distance is ALWAYS positive
    x_start, y_start, z_start, heading = -(311.2 + distance), 158.9, 162, 89
    spawn_transform = carla.Transform(carla.Location(x=x_start, y=y_start, z=z_start), carla.Rotation(pitch=0, yaw=heading, roll=0))
    vehicle2 = world.spawn_actor(vehicle_blueprint, spawn_transform)
    print('vehicle 2 successfully spawned')

    vehicle1.apply_control(carla.VehicleControl(brake=1.0))
    vehicle2.apply_control(carla.VehicleControl(brake=1.0))

    return [vehicle1, vehicle2]