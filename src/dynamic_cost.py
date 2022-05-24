"""
An optional function for dynamic cost computation must accept the arguments as such:
dynamic_cost_computation(can_move_to_indices, state, grid_unit)
and return an array of scalar cost values for each index in can_move_to_indices
"""
from src.ca_utils import (
    get_indices_of_state,
    get_indices_in_area,
    get_grid_size,
    remove_indices_outside_grid,
)
from src.evolution_utils import (
    on_same_axis,
    extra_diagonal_units,
    get_movable_neighbor_indices,
    euclidean_distance,
)
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from queue import PriorityQueue


def get_diagonal_movement_cost(p_index, can_move_to_indices, grid_unit):
    """
    p_index -> tuple, current pedestrian location index
    can_move_to_indices -> numpy array, dim C,2: first entry is the current p_index
    grid_unit -> float: cell length in meters

    return -> numpy array, same dimension as can_move_to_indices, movement cost for each cell in can_move_to_indices

    special dynamic cost: always used, called in different place, other arguments possible
    """
    n_neighbors = can_move_to_indices.shape[0]
    diagonal_cost = np.zeros(n_neighbors)
    for i in range(n_neighbors):
        if not on_same_axis(p_index, can_move_to_indices[i, :]):
            diagonal_cost[i] += (extra_diagonal_units * grid_unit)
    return diagonal_cost


def generic_repulsion(can_move_to_indices, state, grid_unit, maximum_repulsion_range, repulsion_against_state):
    """
    can_move_to_indices -> numpy array, dim C,2: first entry is the current p_index
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell length in meters
    maximum_repulsion_range -> float: maximum distance for effect of repulsion in meters
    repulsion_against_state -> str: which cell state is repulsing

    return -> numpy array, same dimension as can_move_to_indices, repulsion cost for each cell in can_move_to_indices
    """
    range_grid_units = np.ceil(maximum_repulsion_range / grid_unit)
    # get the effective neighborhood -> don't need to calculate distance to all repulsing cells, only close ones
    upper_left_corner = np.min(can_move_to_indices, axis=0) - range_grid_units
    lower_right_corner = np.max(can_move_to_indices, axis=0) + range_grid_units
    neighborhood = get_indices_in_area(upper_left_corner, lower_right_corner)  # N,2
    neighborhood = remove_indices_outside_grid(neighborhood, get_grid_size(state))  # M,2
    # remove the current index (first entry of can_move_to_indices) -> no auto-repulsion
    neighborhood = neighborhood[~np.all(neighborhood == can_move_to_indices[0], axis=1)]
    repulsing_is = np.nonzero(state[repulsion_against_state][neighborhood[:, 0], neighborhood[:, 1]])[0]  # R,
    if not repulsing_is.size:
        # if no repulsing cell in effective neighborhood: repulsion cost is zero
        return np.zeros(can_move_to_indices.shape[0])
    repulsing_locations = neighborhood[repulsing_is, :]  # R,2
    all_differences = can_move_to_indices[:, None, :] - repulsing_locations[None, :, :]  # C,R,2
    all_distances = np.sqrt(np.sum(all_differences ** 2, axis=-1)) * grid_unit  # in meters, C,R
    repulsion_cost = repulsion_cost_function(all_distances, maximum_repulsion_range)  # C,R
    repulsion_cost = repulsion_cost.sum(axis=-1)  # C,
    return repulsion_cost


def pedestrian_repulsion(can_move_to_indices, state, grid_unit):
    """
    can_move_to_indices -> numpy array, dim C,2: first entry is the current p_index
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell length in meters

    return -> numpy array, same dimension as can_move_to_indices, repulsion cost for each cell in can_move_to_indices

    pedestrian repulsion cost for each cell in can_move_to_indices
    """
    personal_space = 0.8
    return generic_repulsion(can_move_to_indices, state, grid_unit, personal_space, "pedestrian")


def obstacle_repulsion(can_move_to_indices, state, grid_unit):
    """
    can_move_to_indices -> numpy array, dim C,2: first entry is the current p_index
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell length in meters

    return -> numpy array, same dimension as can_move_to_indices, repulsion cost for each cell in can_move_to_indices

    obstacle repulsion cost for each cell in can_move_to_indices
    """
    personal_space = 0.8
    return generic_repulsion(can_move_to_indices, state, grid_unit, personal_space, "pedestrian")


def repulsion_cost_function(distances, max_distance):
    """
    distances -> numpy array of distances
    max_distance -> float: parameter

    return -> numpy array, dim same as distances: repulsion cost for each given distance

    computes the repulsion cost between two cells with the maximum effective repulsion distance of max_distance
    """
    repulsion_cost = np.zeros_like(distances)
    repulsion_mask = distances < max_distance
    repulsion_cost[repulsion_mask] = np.exp(1 / (distances[repulsion_mask] ** 2 - max_distance ** 2))
    return repulsion_cost


def plot_repulsion_function(max_distance, ax=None):
    """
    max_distance -> float: parameter
    ax -> a matplotlib axes object, will plot onto given axes

    return -> matplotlib axes
    plots the repulsion function with the given max_distance parameter
    """
    if ax is None:
        _, ax = plt.subplots()
    x_limit = 3 * max_distance
    distances = np.linspace(0, x_limit, 200)
    repulsion_cost = repulsion_cost_function(distances, max_distance)
    ax.plot(distances, repulsion_cost, "k", label=r"$r_{max}="+str(max_distance)+"$")
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("repulsion cost")
    ax.legend()
    return ax


def current_shortest_path(can_move_to_indices, state, grid_unit):
    """
    can_move_to_indices -> numpy array, dim C,2: first entry is the current p_index
    state -> dict: the self.state variable of the ca model
    grid_unit -> float: cell length in meters

    return -> numpy array, same dimension as can_move_to_indices, path cost for each cell in can_move_to_indices

    returns the shortest path length from each index in can_move_to_indices to any target in the current state
    configuration (eg. also considering pedestrians blocking a path)
    """
    n_neighbors = can_move_to_indices.shape[0]
    distances = np.ones(n_neighbors) * np.inf
    target_indices = get_indices_of_state(state, "target")
    for i in range(n_neighbors):
        neighbor = can_move_to_indices[i, :]
        shortest_distance = search_shortest_path_to_target(neighbor, target_indices, state)
        if shortest_distance is not None:
            distances[i] = shortest_distance
    # transform to correct units
    distances += grid_unit
    return distances


def search_shortest_path_to_target(start_index, stop_indices, state):
    """
    start_index -> tuple len=2, index location of where to start search
    stop_indices -> numpy array, dim N,2: the possible targets where to end
    state -> dict: the self.state variable of the ca model

    return -> float: minumum distance to target, or None, if unreachable

    uniform cost search for shortest path from start_index to any index in stop_indices
    """
    grid_size = (state["empty"].shape[0], state["empty"].shape[1])
    tiebreaker = count()
    unvisited_cells = PriorityQueue()
    unvisited_cells.put((0, next(tiebreaker), start_index))
    visited_cells = set()
    while not unvisited_cells.empty():
        distance, _, current_min_cell = unvisited_cells.get()
        cell_tuple = tuple(current_min_cell)
        # check if we reached the goal
        if np.any(np.all(current_min_cell[None, :] == stop_indices, axis=1)):
            return distance
        # check if we explored this cell before
        if cell_tuple in visited_cells:
            continue
        # visit the cell: add to visited
        visited_cells.add(cell_tuple)
        # get the cells neighbors and loop over them
        neighbors = get_movable_neighbor_indices(state, current_min_cell, grid_size, absorbing=True)
        for i in range(neighbors.shape[0]):
            neighbor = neighbors[i, :]
            # if we have explored this neighbor, continue to next
            if tuple(neighbor) in visited_cells:
                continue
            # else, get the distance to the neighbor and add to the priority queue of unvisited cells
            new_distance = euclidean_distance(neighbor, current_min_cell) + distance
            unvisited_cells.put((new_distance, next(tiebreaker), neighbor))
    return None
