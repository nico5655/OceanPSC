#!/usr/bin/python3

# Semi-phisically-based hydraulic erosion simulation. Code is inspired by the
# code found here:
#   http://ranmantaru.com/blog/2011/10/08/water-erosion-on-heightmap-terrain/
# With some theoretical inspiration from here:
#   https://hal.inria.fr/inria-00402079/document

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import utils
from time import time


# Smooths out slopes of `terrain` that are too steep. Rough approximation of the
# phenomenon described here: https://en.wikipedia.org/wiki/Angle_of_repose


def apply_slippage(terrain, repose_slope, cell_width):
    delta = utils.gaussian_gradient(terrain) / cell_width
    smoothed = utils.gaussian_blur(terrain, sigma=1.5)
    should_smooth = np.abs(delta) > repose_slope
    result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
    return result


# Grid constants
full_width = 1000
dim = 128
shape = [dim] * 2
cell_width = full_width / dim
cell_area = cell_width ** 2

x = np.linspace(19*full_width, 20*full_width, dim)
y = np.linspace(19*full_width, 20*full_width, dim)
X, Y = np.meshgrid(x, y)
terrain = 600*utils.f(X/2000, Y/2000)+600

# Snapshotting parameters. Only needed for generating the simulation
# timelapse.
enable_snapshotting = True
my_dir = r"../"
snapshot_dir = os.path.join(my_dir, 'sim_snaps')
snapshot_file_template = 'sim-%05d.png'
if enable_snapshotting:
    try:
        os.mkdir(snapshot_dir)
    except:
        pass


def erosion(terrain, enable_snapshotting=False):

    dim=terrain.shape[0]
    shape=terrain.shape
    cell_width = full_width / dim 
    cell_area = cell_width ** 2
    x = np.linspace(19*full_width, 20*full_width, dim)
    y = np.linspace(19*full_width, 20*full_width, dim)
    X, Y = np.meshgrid(x, y)

    Min = np.ndarray.min(terrain)
    Max = np.ndarray.max(terrain)
     
    terrain = utils.normalize(terrain)

    # Water-related constants
    rain_rate = 0.0008 * cell_area
    evaporation_rate = 0.0005

    # Slope constants
    min_height_delta = 0.05
    repose_slope = 0.1
    gravity = 30.0
    gradient_sigma = 0.5

    # Sediment constants
    sediment_capacity_constant = 50
    dissolving_rate = 0.25
    deposition_rate = 0.001

    # The numer of iterations is proportional to the grid dimension. This is to
    # allow changes on one side of the grid to affect the other side.
    iterations = int(0.4 * dim)

    # `terrain` represents the actual terrain height we're interested in
    # terrain = utils.fbm(shape, -2.0)
    # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
    # transfered to/from sediment depending on a number of different factors.

    sediment = np.zeros_like(terrain)

    # The amount of water. Responsible for carrying sediment.
    water = np.zeros_like(terrain)

    # The water velocity.
    velocity = np.zeros_like(terrain)
    for i in range(0, iterations):

        # Add precipitation. This is done by via simple uniform random distribution,
        # although other models use a raindrop model
        water += np.random.rand(*shape) * rain_rate

        # Compute the normalized gradient of the terrain height to determine where
        # water and sediment will be moving.
        gradient = utils.simple_gradient(terrain)
        gradient = np.select([np.abs(gradient) < 1e-10],
                             [np.exp(2j * np.pi * np.random.rand(*shape))],
                             gradient)
        gradient /= np.abs(gradient)

        # Compute the difference between the current height and the height offset by
        # `gradient`.
        neighbor_height = utils.sample(terrain, -gradient)
        height_delta = terrain - neighbor_height

        # The sediment capacity represents how much sediment can be suspended in
        # water. If the sediment exceeds the quantity, then it is deposited,
        # otherwise terrain is eroded.
        sediment_capacity = (
            (np.maximum(height_delta, min_height_delta) / cell_width) * velocity *
            water * sediment_capacity_constant)
        deposited_sediment = np.select(
            [
                height_delta < 0,
                sediment > sediment_capacity,
            ], [
                np.minimum(height_delta, sediment),
                deposition_rate * (sediment - sediment_capacity),
            ],
            # If sediment <= sediment_capacity
            dissolving_rate * (sediment - sediment_capacity))

        # Don't erode more sediment than the current terrain height.
        deposited_sediment = np.maximum(-height_delta, deposited_sediment)

        # Update terrain and sediment quantities.
        sediment -= deposited_sediment
        terrain += deposited_sediment
        sediment = utils.displace(sediment, gradient)
        water = utils.displace(water, gradient)

        # Smooth out steep slopes.
        terrain = apply_slippage(terrain, repose_slope, cell_width)

        # Update velocity
        velocity = gravity * height_delta / cell_width

        # Apply evaporation
        water *= 1 - evaporation_rate

        # Snapshot, if applicable.
        if enable_snapshotting:
            output_path = os.path.join(
                snapshot_dir, snapshot_file_template % i)
            utils.save_as_png(utils.normalize(terrain), output_path)

            np.save(
                r"../"+str(i+1), Min + terrain*(Max-Min))
    terrain = Min + terrain*(Max-Min)

    
    output_path = os.path.join(
        snapshot_dir, "sim-final.png")
    utils.save_as_png(utils.normalize(terrain), output_path)
    print(terrain.shape,np.max(terrain),np.min(terrain),np.max(terrain)-np.min(terrain))
    return terrain

