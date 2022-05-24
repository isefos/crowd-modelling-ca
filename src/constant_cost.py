"""
functions that pre-compute a cost map for the given state, should take arguments:
constant_cost_computation(state, grid_unit)
and return a 2D array of the same size as the state grid with a scalar cost for each cell in the grid
"""
from src.ca_utils import (
    get_indices_of_state,
    get_grid_size,
)
from src.evolution_utils import (
    get_movable_neighbor_indices,
    euclidean_distance,
)
import numpy as np
from itertools import count
from queue import PriorityQueue


def uniform_cost_map(state, grid_unit):
    """
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell unit size in meters

    return -> state, with the newly set constant_cost_map
    """
    state["constant_cost_map"] = np.zeros_like(state["empty"], dtype=float)
    obstacle_indices = get_indices_of_state(state, "obstacle")  # N,2
    state["constant_cost_map"][obstacle_indices[:, 0], obstacle_indices[:, 1]] = np.inf
    return state


def min_euclidean_distance_no_obstacle_avoidance_cost_map(state, grid_unit):
    """
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell unit size in meters

    return -> state, with the newly set constant_cost_map
    """
    all_indices = np.indices(get_grid_size(state))[:, :, :, None]  # 2,X,Y,1
    target_indices = get_indices_of_state(state, "target").T[:, None, None, :]  # 2,1,1,N
    euclidean_distances = np.sqrt(np.sum((all_indices - target_indices) ** 2, axis=0))  # 2,X,Y,T -sum-> X,Y,N
    min_euclidean_distance_to_target = np.min(euclidean_distances, axis=-1)  # X,Y
    state["constant_cost_map"] = min_euclidean_distance_to_target
    # transform to physical units
    state["constant_cost_map"] *= grid_unit
    return state


def min_euclidean_distance_cost_map(state, grid_unit):
    """
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell unit size in meters

    return -> state, with the newly set constant_cost_map
    """
    state = min_euclidean_distance_no_obstacle_avoidance_cost_map(state, grid_unit)
    # set obstacles to infinite
    obstacle_indices = get_indices_of_state(state, "obstacle")  # M,2
    state["constant_cost_map"][obstacle_indices[:, 0], obstacle_indices[:, 1]] = np.inf
    return state


def shortest_path_cost_map(state, grid_unit):
    """
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell unit size in meters

    return -> state, with the newly set constant_cost_map
    """
    state["constant_cost_map"] = -1 * np.ones_like(state["empty"], dtype=float)
    # set obstacles to infinite
    obstacle_indices = get_indices_of_state(state, "obstacle")  # N,2
    state["constant_cost_map"][obstacle_indices[:, 0], obstacle_indices[:, 1]] = np.inf
    # now flood the cells with the shortest distance values from a target
    target_indices = get_indices_of_state(state, "target")  # M,2
    state, maximum_distance = flood_cells_dijkstra(target_indices, state)
    # values that are unreachable from the targets are still at -1, set them to the maximum value + 1
    # (don't want it to be np.inf, since we still want them to be able to choose between them using other costs)
    x_unreachable, y_unreachable = np.nonzero(state["constant_cost_map"] == -1)
    state["constant_cost_map"][x_unreachable, y_unreachable] = maximum_distance + 1
    # transform to physical units
    state["constant_cost_map"] *= grid_unit
    return state


def flood_cells_dijkstra(t_indices, state):
    """
    state -> dict: the self.state variable of the ca model

    return -> tuple len 2:
    -> state, with the newly set constant_cost_map
    -> float: the maximum path distance from targets to any reachable cell
    """
    grid_size = (state["empty"].shape[0], state["empty"].shape[1])
    tiebreaker = count()
    unvisited_cells = PriorityQueue()
    for i in range(t_indices.shape[0]):
        unvisited_cells.put((0, next(tiebreaker), t_indices[i, :]))
    visited_cells = set()
    maximum_distance = 0
    while not unvisited_cells.empty():
        current_distance, _, current_min_cell = unvisited_cells.get()
        cell_tuple = tuple(current_min_cell)
        # check if we explored this cell before
        if cell_tuple in visited_cells:
            continue
        # visit the cell: add to visited and set the distance in the cost map
        visited_cells.add(cell_tuple)
        state["constant_cost_map"][current_min_cell[0], current_min_cell[1]] = current_distance
        maximum_distance = current_distance
        # get the cells neighbors and loop over them
        neighbors = get_movable_neighbor_indices(state, current_min_cell, grid_size,
                                                 absorbing=False, include_pedestrians=True)
        for i in range(neighbors.shape[0]):
            neighbor = neighbors[i, :]
            # if we have explored this neighbor, continue to next
            if tuple(neighbor) in visited_cells:
                continue
            # else, get the distance to the neighbor and add to the priority queue of unvisited cells
            new_distance = euclidean_distance(neighbor, current_min_cell) + current_distance
            unvisited_cells.put((new_distance, next(tiebreaker), neighbor))
    return state, maximum_distance
