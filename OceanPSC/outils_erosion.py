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
    result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
    return result




def precipitation_erosion(terrain, scale,rain_frequency=0.0008,
evaporation_rate = 0.0005,sediment_capacity_constant = 50,dissolving_rate = 0.25 ,deposition_rate = 0.001):

    
    shape = terrain.shape
    dim=shape[0]
    cell_width = scale
    cell_area = cell_width ** 2

    Min = np.ndarray.min(terrain)
    Max = np.ndarray.max(terrain)

    terrain = utils.normalize(terrain)

    rain_rate = rain_frequency * cell_area
    

    # Slope constants
    min_height_delta = 0.05
    repose_slope = 0.03
    gravity = 30.0
    
    

    # The numer of iterations is proportional to the grid dimension. This is to
    # allow changes on one side of the grid to affect the other side.
    iterations = int(1.4 * dim)

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
        # -----------------------------------------
        water += np.random.rand(*shape) * rain_rate

        # Compute the normalized gradient of the terrain height to determine where
        # water and sediment will be moving.
        gradient = np.zeros_like(terrain, dtype='complex')
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

      
    terrain = Min + terrain*(Max-Min)
    return terrain

def erosion(terrain, scale,water_map,
evaporation_rate = 0.0005,sediment_capacity_constant = 50,dissolving_rate = 0.25 ,deposition_rate = 0.001,iterations=50):

    
    shape = terrain.shape
    dim=shape[0]
    cell_width = scale
    cell_area = cell_width ** 2

    Min = np.ndarray.min(terrain)
    Max = np.ndarray.max(terrain)

    terrain = utils.normalize(terrain)
    

    # Slope constants
    min_height_delta = 0.05
    repose_slope = 0.03
    gravity = 30.0
    
    

    

    # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
    # transfered to/from sediment depending on a number of different factors.
    water=np.zeros_like(terrain)
    sediment = np.zeros_like(terrain)

    # The amount of water. Responsible for carrying sediment.
    # The water velocity.
    velocity = np.zeros_like(terrain)
    for i in range(0, iterations):

        water+=water_map
        # Compute the normalized gradient of the terrain height to determine where
        # water and sediment will be moving.
        gradient = np.zeros_like(terrain, dtype='complex')
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

      
    terrain = Min + terrain*(Max-Min)
    return terrain


    
