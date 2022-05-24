from src.ca_utils import (
    set_cell_state,
    remove_indices_outside_grid,
)
import numpy as np


neighbor_square_3x3_offsets = np.array(
    [
        [-1, -1, -1,  0,  0,  1,  1,  1],  # rows
        [-1,  0,  1, -1,  1, -1,  0,  1]  # columns
    ]
)
non_diagonal_neighbor_offsets = np.array(
    [
        [-1,  0,  0,  1],  # rows
        [ 0, -1,  1,  0]  # columns
    ]
)
extra_diagonal_units = np.sqrt(2) - 1
rng = np.random.default_rng()


def move_or_not(p):
    """
    p -> float between 0 and 1

    return -> bool

    return True with probability p
    """
    return p > rng.random()


def euclidean_distance(cell_index_1, cell_index_2):
    """
    cell_index_1 -> tuple: (height, width)
    cell_index_2 -> tuple: (height, width)

    return -> float
    """
    return np.sqrt(np.sum((cell_index_1 - cell_index_2)**2))


def get_all_euclidean_distances(from_indices, to_indices):
    """
    from_indices -> numpy array, dim F,2
    to_indices -> numpy array, dim T,2

    return -> numpy array, dim F,T
    """
    all_differences = from_indices[:, None, :] - to_indices[None, :, :]  # F,T,2
    return np.sqrt(np.sum(all_differences ** 2, axis=2))  # F,T


def get_movable_neighbor_indices(
        state,
        p_index,
        grid_size,
        absorbing,
        include_pedestrians=False,
        include_obstacles=False,
):
    """
    state -> dict: the self.state variable of the ca model
    p_index -> tuple, current pedestrian location index
    grid_size -> tuple: of 2 ints, the model grid dimensions in number of cells
    absorbing -> bool
    include_pedestrians -> bool
    include_obstacles -> bool

    return -> numpy array, dim M,2

    get all the neighbor indices the pedestrian can move to:
     - its own (stay put)
     - the empty neighbors (except for those that are in the diagonal, next to an obstacle)
     - if absorbing=True -> also the targets
    """
    movable = []
    neighbors = remove_indices_outside_grid(p_index + neighbor_square_3x3_offsets.T, grid_size)
    non_diagonal_neighbors = remove_indices_outside_grid(p_index + non_diagonal_neighbor_offsets.T, grid_size)
    for add_bool, state_type in [(True, "empty"),
                                 (absorbing, "target"),
                                 (include_pedestrians, "pedestrian"),
                                 (include_obstacles, "obstacle")]:
        if not add_bool:
            continue
        check = boolean_mask_check_indices_state(state, neighbors, state_type)
        movable.append(neighbors[check, :])  # N_i,2
    movable = np.vstack(movable)  # N,2
    # if present, remove indices that are in the diagonal next to an obstacle -> no movement through corners
    obstacle_check = boolean_mask_check_indices_state(state, non_diagonal_neighbors, "obstacle")
    non_diagonal_obstacles = non_diagonal_neighbors[obstacle_check, :]  # M,2
    # if we include obstacles, we also don't care about the corners
    if (not include_obstacles) and non_diagonal_obstacles.size and movable.size:
        next_to_obstacle_check = boolean_mask_check_if_next_to(movable, non_diagonal_obstacles)
        movable = movable[~next_to_obstacle_check]
    # permute the neighbors, such that they are not always in the same order
    movable = np.random.permutation(movable)
    # add the current p_index as well (can stay put)
    movable = np.vstack((p_index[None, :], movable))  # N',2
    return movable


def boolean_mask_check_indices_state(state, indices, state_to_check):
    """
    state -> dict: the self.state variable of the ca model
    indices -> numpy array dim N,2
    state_to_check -> str: wich state to check for

    returns -> numpy array: boolean mask dim N,

    True at index i if indices[i] is of state state_to_check, False otherwise
    """
    return np.nonzero(state[state_to_check][indices[:, 0], indices[:, 1]])[0]


def boolean_mask_check_if_next_to(to_check, possible_neighbors):
    """
    to_check -> numpy array dim N,2
    possible_neighbors -> numpy array dim M,2

    return -> numpy array boolean mask dim N,

    for all the to_check indices (N,2) checking whether they are directly next to of any indices in
    possible_neighbors (M,2). (Directly next to -> non-diagonal neighbor)
    """
    distances = get_all_euclidean_distances(to_check, possible_neighbors)  # N,M
    return np.any((distances == 1), axis=1)  # N,


def on_same_axis(index_from, index_to):
    """
    index_from -> tuple len=2, location index
    index_to -> tuple len=2, location index

    return -> bool

    checks whether the move leaves the index in at least one axis the same (no diagonal movement)
    """
    if (index_from[0] == index_to[0]) or (index_from[1] == index_to[1]):
        return True
    return False


def move_pedestrian(state, old_p_index, new_p_index, no_obstacle_avoidance=False):
    """
    state -> dict: the self.state variable of the ca model
    old_p_index -> tuple len=2, location index
    new_p_index -> tuple len=2, location index
    no_obstacle_avoidance -> bool

    return -> state, as modified

    moves a pedestrian cell from one index to another index.
    """
    try:
        assert state["pedestrian"][old_p_index[0], old_p_index[1]]
        # always check that the cell we are moving to is empty. Unless no obstacle avoidance is used (intentionally
        # breaking the simulation)
        if not no_obstacle_avoidance:
            assert state["empty"][new_p_index[0], new_p_index[1]]
    except AssertionError:
        raise Exception(f"Tried illegal move, please debug!")
    # get the pedestrian id
    p_id = state["pedestrian_id"][old_p_index[0], old_p_index[1]]
    # if the move is diagonal, add the extra diagonal units to the pedestrian attributes
    if not on_same_axis(old_p_index, new_p_index):
        state["pedestrian_attributes"][p_id]["additional_diag_units"] += extra_diagonal_units
    # set old index to empty
    state = set_cell_state(state, old_p_index, "empty")
    # set new index to pedestrian, and the pedestrian id to the new location
    state = set_cell_state(state, new_p_index, "pedestrian")
    state["pedestrian_id"][new_p_index[0], new_p_index[1]] = p_id
    return state


def absorb_pedestrian(state, p_index, t_index):
    """
    state -> dict: the self.state variable of the ca model
    p_index -> tuple len=2, location index
    t_index -> tuple len=2, location index

    return -> state, as modified

    moves a pedestrian cell from old_p_index into the target at new_p_index.
    simply set the old_p_index to "empty" and give the target cell the pedestrian_id, so that we can track into which
    target cell it was absorbed
    """
    try:
        assert state["pedestrian"][p_index[0], p_index[1]]
        assert state["target"][t_index[0], t_index[1]]
    except AssertionError:
        raise Exception(f"Tried illegal move, please debug!")
    # transfer the pedestrian id into the target
    p_id = state["pedestrian_id"][p_index[0], p_index[1]]
    # set old index to empty
    state = set_cell_state(state, p_index, "empty")
    state["pedestrian_id"][t_index[0], t_index[1]] = p_id
    return state
