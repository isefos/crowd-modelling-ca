"""
file contains mostly useful functions to manipulate the state of the ca model
"""
import numpy as np
from matplotlib.widgets import Slider


def get_grid_size(state):
    """
    state -> dict: the self.state variable of the ca model

    return -> tuple: the dimensions of the model grid
    """
    grid_size = (state["empty"].shape[0], state["empty"].shape[1])
    return grid_size


def remove_indices_outside_grid(indices, grid_size):
    """
    indices -> numpy array, dim N,2
    grid_size -> tuple: of 2 ints, the model grid dimensions in number of cells

    return -> numpy array. dim M,2
    removes cells in indices that aren't inside the valid grid
    """
    # make sure the grid size is not violated
    valid_rows = np.nonzero(np.logical_and(indices[:, 0] >= 0, indices[:, 0] < grid_size[0]))[0]
    valid_cols = np.nonzero(np.logical_and(indices[:, 1] >= 0, indices[:, 1] < grid_size[1]))[0]
    # only those with both valid rows and columns are completely valid -> intersection
    valid = np.intersect1d(valid_rows, valid_cols)
    # return only the valid neighbors
    return indices[valid, :]


delete_states = {"empty", "pedestrian", "obstacle", "target", "pedestrian_id"}


def set_cell_state(state, cell_loc=(0, 0), set_to="empty"):
    """
    state -> dict: the self.state variable of the ca model
    cell_loc -> tuple, (height, width)
    set_to -> str: must be a valid settable state

    return -> state, with the newly set state

    change the cell state to one of: "empty", "pedestrian", "obstacle", or "target"

    -> don't use directly for pedestrian, because it requires extra care: move/generate pedestrian_id as well, and add
       the needed attributes
    """
    for key, value in state.items():
        if key == set_to:
            value[cell_loc[0], cell_loc[1]] = 1
        elif key in delete_states:
            value[cell_loc[0], cell_loc[1]] = 0
    return state


def get_cells_in_area(state, upper_left_corner=(0, 0), lower_right_corner=(1, 1)):
    """
    state -> dict: the self.state variable of the ca model
    upper_left_corner -> tuple: (height, width)
    lower_right_corner -> tuple: (height, width)

    return -> numpy array, dim N,2 contains all the indices in the rectangular area defined by the corners, additionally
     checks if area is valid for given grid size (determined via state)
    """
    grid_size = get_grid_size(state)
    assert np.all(upper_left_corner >= (0, 0))
    assert np.all(lower_right_corner < grid_size)
    return get_indices_in_area(upper_left_corner, lower_right_corner)


def set_cells_in_area_state(state, upper_left_corner=(0, 0), lower_right_corner=(1, 1), set_to="empty"):
    """
    state -> dict: the self.state variable of the ca model
    upper_left_corner -> tuple: (height, width)
    lower_right_corner -> tuple: (height, width)
    set_to -> str: must be a valid settable state

    return: state, with the newly set states
    """
    set_cells_indices = get_cells_in_area(state, upper_left_corner, lower_right_corner)
    for i in range(set_cells_indices.shape[0]):
        state = set_cell_state(state, set_cells_indices[i, :], set_to)
    return state


def set_cell_pedestrian(
        state,
        simulation_parameters,
        cell_loc=(0, 0),
        pedestrian_id=None,
        additional_diag_units=0,
        evacuation_time=None,
        age=None,
        sex=None,
        speed_m_per_s=None,
        max_speed_fraction=None,
):
    """
    state -> dict: the self.state variable of the ca model
    simulation_parameters -> obj of class SimulationParameters
    cell_loc -> tuple: (height, width)
    pedestrian_id -> int: ID
    age -> float: set a given age
    sex -> str: must be either 'female' or 'male'
      Only set at most one of the following:
    speed_m_per_s -> float: sets the desired speed in m/s
    max_speed_fraction -> float: between 0 and 1, sets the speed as a fraction of the maximum speed

    return: state, with the newly set pedestrian state
    """
    try:
        assert speed_m_per_s is None or max_speed_fraction is None
    except AssertionError:
        raise Exception("Please only specify either the pedestrian speed in meters per second, or as a percentage of "
                        "the maximum speed, but not both!")
    for key, value in state.items():
        if key == "pedestrian":
            value[cell_loc[0], cell_loc[1]] = 1
        elif key == "pedestrian_id":
            value[cell_loc[0], cell_loc[1]] = pedestrian_id
        elif key == "pedestrian_attributes":
            # fill in the attribute dictionary of this particular pedestrian (with pedestrian_id)
            value[pedestrian_id] = {"additional_diag_units": additional_diag_units, "evacuation_time": evacuation_time}
            if age is None:
                # sample age if none is given
                age = simulation_parameters.get_age_from_distribution()
            else:
                # if an age is given, make sure it is in the possible range
                assert simulation_parameters.age_min <= age <= simulation_parameters.age_max
            value[pedestrian_id]["age"] = age
            if sex is None:
                # sample a sex if none is given
                sex = simulation_parameters.get_sex_from_distribution()
            else:
                # make sure given age is valid
                assert sex in ["female", "male"]
            value[pedestrian_id]["sex"] = sex
            if max_speed_fraction is None and speed_m_per_s is None:
                # if no speed is given, sample it
                speed_m_per_s = simulation_parameters.get_speed_from_distribution(age, sex)
                max_speed_fraction = simulation_parameters.get_max_speed_fraction(speed_m_per_s)
            # if speed is given either in m/s or in fraction of max_speed convert it to the other (range checks are
            # already included inside the conversion methods).
            elif max_speed_fraction is None:
                max_speed_fraction = simulation_parameters.get_max_speed_fraction(speed_m_per_s)
            else:
                speed_m_per_s = simulation_parameters.get_speed_from_fraction(max_speed_fraction)
            value[pedestrian_id]["speed"] = speed_m_per_s
            value[pedestrian_id]["max_speed_fraction"] = max_speed_fraction
        elif key in delete_states:
            value[cell_loc[0], cell_loc[1]] = 0
    return state


def get_indices_in_area(upper_left_index, lower_right_index):
    """
    upper_left_corner -> tuple: (height, width)
    lower_right_corner -> tuple: (height, width)

    return -> numpy array, dim N,2 contains all the indices in the rectangular area defined by the corners
    """
    assert np.all(upper_left_index <= lower_right_index)
    a = np.meshgrid(np.arange(upper_left_index[0], lower_right_index[0] + 1, dtype=int),
                    np.arange(upper_left_index[1], lower_right_index[1] + 1, dtype=int))
    return np.vstack([np.hstack(i) for i in a]).T  # N,2


def get_all_indices(state):
    """
    state -> dict: the self.state variable of the ca model

    return -> numpy array, dim N,2 contains all possible indices in grid
    """
    grid_size = get_grid_size(state)
    return get_indices_in_area((0, 0), (grid_size[0] - 1, grid_size[1] - 1))


def get_indices_of_state(state, cell_type="empty"):
    """
   state -> dict: the self.state variable of the ca model

   return -> numpy array, dim N,2 contains all the cell indices in the grid that are of the given state
   """
    return np.stack(np.nonzero(state[cell_type][:, :])).T


def get_plot_alphas(plot_data):
    """
    plot_data -> numpy array: as the one produced by ca_model._get_plot_data()

    return -> plot_data -> numpy array: same dimension as input, boolean mask to mask out empty cells
    """
    return (~(plot_data == 0)).astype(float)


def plot_paths(paths, ax, steps=0):
    """
    paths -> dict {p_id: np.array T,}
    ax -> matplotlib axes: plots onto ax
    steps -> int: plot the lines up to how many steps

    return -> dict of matplotlib line objects, returned such that interactive plot can update the data easily
    """
    lines = dict()
    for p_id, path in paths.items():
        line, = ax.plot(path[1, :steps+1] + 0.5, path[0, :steps+1] + 0.5, "k", alpha=0.07, zorder=1)
        lines[p_id] = line
    return lines


def create_step_slider(
        fig,
        im,
        ax_step_slider,
        plots,
        paths,
        lines,
        time_step,
):
    """
    fig -> matplotlib figure to which to add slider
    im -> matplotlib image object created by ca._plot_data(), to update the data with slider
    ax_step_slider -> matplotlib axes, where to place the slider on figure
    plots -> numpy array, dim S,H,W: containing all the saved plot data during simulation
    paths -> dict: for each ID the numpy array of the simulation path walked, dim 2,S
    lines -> dict of matplotlib line objects per ID
    time_step -> float, the time step in seconds, to scale the axis correctly in seconds

    return -> matplotlib widget slider object
    """
    steps = plots.shape[0]
    step_slider = Slider(ax_step_slider, "Seconds", 0, steps * time_step, valinit=0, initcolor='none')

    def update_to_second(val):
        # get the time step index of the second
        step = int(np.floor(step_slider.val / time_step))
        for p_id, line in lines.items():
            line.set_data(paths[p_id][1, :step+1] + 0.5, paths[p_id][0, :step+1] + 0.5)
        im.set(data=plots[step, :, :], alpha=get_plot_alphas(plots[step, :, :]))
        fig.canvas.draw_idle()

    step_slider.on_changed(update_to_second)
    return step_slider


def add_meter_units_to_plot(ax, grid_size, grid_unit):
    """
    ax -> matplotlib axes, which to add the meter ticks to
    grid_size -> tuple: (height, width)
    grid_unit -> float: cell unit size in meters

    adds meter ticks to the state map plot
    """
    max_number_of_ticks = 10
    y_units, x_units = grid_size
    x_meters = int(np.floor(x_units * grid_unit))
    y_meters = int(np.floor(y_units * grid_unit))
    tick_meters = max(np.ceil(x_meters / max_number_of_ticks), np.ceil(y_meters / max_number_of_ticks))
    x_tick_locations = np.arange(start=0, stop=x_units + 1, step=tick_meters * (1 / grid_unit))
    y_tick_locations = np.arange(start=0, stop=y_units + 1, step=tick_meters * (1 / grid_unit))
    x_tick_labels = np.arange(start=0, stop=x_meters + 1, step=tick_meters, dtype=int)
    y_tick_labels = np.arange(start=0, stop=y_meters + 1, step=tick_meters, dtype=int)
    ax.set_xticks(x_tick_locations, labels=x_tick_labels)
    ax.set_yticks(y_tick_locations, labels=y_tick_labels)
    ax.set_xlabel("[m]")
    ax.set_ylabel("[m]")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
