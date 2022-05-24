"""
CA: Cellular Automaton

contains the main class for modelling crowds with cellular automaton
"""
from src.simulation_parameters import SimulationParameters
from src.ca_utils import (
    set_cell_state,
    set_cells_in_area_state,
    set_cell_pedestrian,
    get_cells_in_area,
    get_indices_of_state,
    add_meter_units_to_plot,
    get_plot_alphas,
    plot_paths,
    create_step_slider,
)
from src.constant_cost import (
    min_euclidean_distance_no_obstacle_avoidance_cost_map,
    uniform_cost_map,
    min_euclidean_distance_cost_map,
    shortest_path_cost_map,
)
from src.dynamic_cost import (
    get_diagonal_movement_cost,
    pedestrian_repulsion,
    obstacle_repulsion,
    current_shortest_path,
)
from src.evolution_utils import (
    move_or_not,
    get_movable_neighbor_indices,
    move_pedestrian,
    absorb_pedestrian,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation


class CrowdModelCellularAutomaton:
    """
    Cellular Automation model for crowd simulations
    """
    constant_cost_computation_options = {
        "no_obstacle_avoidance": min_euclidean_distance_no_obstacle_avoidance_cost_map,
        "uniform": uniform_cost_map,
        "min_euclidean_distance": min_euclidean_distance_cost_map,
        "shortest_path": shortest_path_cost_map
    }
    dynamic_cost_computation_options = {
        "pedestrian_repulsion": pedestrian_repulsion,
        "obstacle_repulsion": obstacle_repulsion,
        "current_shortest_path": current_shortest_path
    }

    def __init__(
        self,
        grid_size=(50, 50),
        constant_cost_map="shortest_path",
        dynamic_costs=("pedestrian_repulsion", ),
        absorbing_targets=True,
        no_obstacle_avoidance=False,
        **simulation_parameters,
    ):
        """
        grid_size -> tuple: model grid size as number of cells, (height, width)
        constant_cost_map -> str: sets the option for how to compute the constant cost map, possible values:
            "no_obstacle_avoidance", "uniform", "min_euclidean_distance", "shortest_path"
        dynamic_costs -> tuple: sets the options (multiple possible in tuple) for how to compute the dynamic cost,
            possible values: "pedestrian_repulsion", "obstacle_repulsion", "current_shortest_path"
        absorbing_targets -> bool: controls whether targets can absorb pedestrians or not
        no_obstacle_avoidance -> bool: used only as a test, to see what happens when obstacle avoidance is deactivated
        **simulation_parameters -> keyword arguments for the SimulationParameters class
        """
        self.simulation_parameters = SimulationParameters(**simulation_parameters)
        assert len(grid_size) == 2
        assert isinstance(grid_size[0], int) and isinstance(grid_size[1], int)
        assert grid_size[0] > 0 and grid_size[1] > 0
        self.grid_size = grid_size
        assert constant_cost_map in self.constant_cost_computation_options
        self.constant_cost_map_computation = self.constant_cost_computation_options[constant_cost_map]
        self.dynamic_cost_computations = []
        for dynamic_cost in dynamic_costs:
            assert dynamic_cost in self.dynamic_cost_computation_options
            self.dynamic_cost_computations.append(self.dynamic_cost_computation_options[dynamic_cost])
        self.absorbing = absorbing_targets
        self.no_obstacle_avoidance = no_obstacle_avoidance
        self.state = {
            "empty": np.ones(grid_size, dtype=np.int8),  # initialize all as empty
            "pedestrian": np.zeros(grid_size, dtype=np.int8),
            "obstacle": np.zeros(grid_size, dtype=np.int8),
            "target": np.zeros(grid_size, dtype=np.int8),
            "pedestrian_id": np.zeros(grid_size, dtype=np.uint32),  # used to track the pedestrians
            "pedestrian_attributes": dict(),  # used to store all the necessary attributes of each pedestrian
            "constant_cost_map": np.zeros(grid_size),  # stores the precomputed cost for each grid cell
        }
        self.pedestrian_id_counter = 1
        self.saved_state = None
        self.plot_color_map = ListedColormap(["white", "darkorange", "black", "palegreen"])
        self.cost_color_map = get_cmap("Reds")
        self.cost_color_map.set_bad(color="k")
        self.simulation_steps = 0  # how many time-steps the simulation has run for
        self.simulation_seconds = 0  # how many seconds the simulation has run for
        self.simulation_plots = None  # T,grid_x,grid_y array, one map plot for each time_step
        self.simulation_paths = dict()  # for P pedestrians: paths -> the 2-dim location at each T time steps
        self.simulation_total_evacuation_time = None  # saves the time at which all the pedestrians were absorbed
        # interactive matplotlib instances need to keep references to stay responsive -> save to object:
        self.step_sliders = []

    def set_cell_empty(self, cell_loc=(0, 0)):
        """
        cell_loc -> tuple: (height, width)

        sets cell at specified location to empty state
        """
        self.state = set_cell_state(self.state, cell_loc, "empty")

    def set_area_empty(self, upper_left_corner, lower_right_corner):
        """
        upper_left_corner -> tuple: (height, width)
        lower_right_corner -> tuple: (height, width)

        sets all cells in the rectangular area defined by its corners to empty state
        """
        self.state = set_cells_in_area_state(self.state, upper_left_corner, lower_right_corner, "empty")

    def set_cell_obstacle(self, cell_loc=(0, 0)):
        """
        cell_loc -> tuple: (height, width)

        sets cell at specified location to obstacle state
        """
        self.state = set_cell_state(self.state, cell_loc, "obstacle")

    def set_area_obstacle(self, upper_left_corner, lower_right_corner):
        """
        upper_left_corner -> tuple: (height, width)
        lower_right_corner -> tuple: (height, width)

        sets all cells in the rectangular area defined by its corners to obstacle state
        """
        self.state = set_cells_in_area_state(self.state, upper_left_corner, lower_right_corner, "obstacle")

    def set_cell_target(self, cell_loc=(0, 0)):
        """
        cell_loc -> tuple: (height, width)

        sets cell at specified location to target state
        """
        self.state = set_cell_state(self.state, cell_loc, "target")

    def set_area_target(self, upper_left_corner, lower_right_corner):
        """
        upper_left_corner -> tuple: (height, width)
        lower_right_corner -> tuple: (height, width)

        sets all cells in the rectangular area defined by its corners to target state
        """
        self.state = set_cells_in_area_state(self.state, upper_left_corner, lower_right_corner, "target")

    def set_cell_pedestrian(
            self,
            cell_loc=(0, 0),
            age=None,
            sex=None,
            speed_m_per_s=None,
            max_speed_fraction=None,
    ):
        """
        cell_loc -> tuple: (height, width)
        age -> float: set a given age
        sex -> str: must be either 'female' or 'male'
          Only set at most one of the following:
        speed_m_per_s -> float: sets the desired speed in m/s
        max_speed_fraction -> float: between 0 and 1, sets the speed as a fraction of the maximum speed

        sets cell at specified location to pedestrian state, with all the given attributes (if not given, they are
        randomly sampled later)
        """
        set_cell_pedestrian(
            self.state,
            self.simulation_parameters,
            cell_loc,
            self.pedestrian_id_counter,
            additional_diag_units=0,
            age=age,
            sex=sex,
            speed_m_per_s=speed_m_per_s,
            max_speed_fraction=max_speed_fraction,
        )
        self.pedestrian_id_counter += 1

    def set_area_pedestrian_random_n(
            self,
            n,
            upper_left_corner,
            lower_right_corner,
            age=None,
            sex=None,
            speed_m_per_s=None,
            max_speed_fraction=None,
    ):
        """
        n -> int: number of pedestrians to place in area
        upper_left_corner -> tuple: (height, width)
        lower_right_corner -> tuple: (height, width)
          The following will apply to all n pedestrians:
        age -> float: set a given age
        sex -> str: must be either 'female' or 'male'
          Only set at most one of the following:
        speed_m_per_s -> float: sets the desired speed in m/s
        max_speed_fraction -> float: between 0 and 1, sets the speed as a fraction of the maximum speed

        sets n pedestrians randomly in the rectangular area defined by the upper_left_corner and lower_right_corner
        """
        area_indices = get_cells_in_area(self.state, upper_left_corner, lower_right_corner)
        n_area = area_indices.shape[0]
        if n_area < n:
            raise Exception(f"Area not large enough to fit {n} pedestrians!")
        select_random_n = np.random.permutation(np.arange(n_area, dtype=int))[:n]
        set_cells_to_pedestrian = area_indices[select_random_n, :]
        for i in range(n):
            self.set_cell_pedestrian(
                cell_loc=set_cells_to_pedestrian[i, :],
                age=age,
                sex=sex,
                speed_m_per_s=speed_m_per_s,
                max_speed_fraction=max_speed_fraction,
            )

    def set_area_pedestrian_random_density(
            self,
            density,
            upper_left_corner,
            lower_right_corner,
            age=None,
            sex=None,
            speed_m_per_s=None,
            max_speed_fraction=None,
    ):
        """
        density -> float: number of pedestrians per square meter [m^2]
            (constrained by an upper bound depending on the simulation parameters)
        upper_left_corner -> tuple: (height, width)
        lower_right_corner -> tuple: (height, width)
          The following will apply to all n pedestrians:
        age -> float: set a given age
        sex -> str: must be either 'female' or 'male'
          Only set at most one of the following:
        speed_m_per_s -> float: sets the desired speed in m/s
        max_speed_fraction -> float: between 0 and 1, sets the speed as a fraction of the maximum speed

        set pedestrians randomly in the area defined by the upper_left_corner and lower_right_corner with the given
        density in pedestrians / m^2
        """
        # convert density from [p/m^2] to [p/u^2], where u is the grid_unit -> multiply by u^2
        density_unit_conversion_scale = self.simulation_parameters.grid_unit ** 2
        grid_unit_density = density * density_unit_conversion_scale
        # check that the grid unit density is in [0, 1], since we cannot simulate a higher density than 1 p per cell
        if not 0 <= grid_unit_density <= 1:
            raise Exception("Given density can not be simulated by this model!")
        area_indices = get_cells_in_area(self.state, upper_left_corner, lower_right_corner)
        n_area = area_indices.shape[0]
        n_pedestrians_set = 0
        for i in range(n_area):
            if np.random.uniform() > grid_unit_density:
                continue
            self.set_cell_pedestrian(
                cell_loc=area_indices[i, :],
                age=age,
                sex=sex,
                speed_m_per_s=speed_m_per_s,
                max_speed_fraction=max_speed_fraction,
            )
            n_pedestrians_set += 1
        actual_grid_unit_density = n_pedestrians_set / n_area
        actual_density = actual_grid_unit_density / density_unit_conversion_scale
        print(f"Randomly set {n_pedestrians_set} pedestrians in the area.\n"
              f"Corresponds to a density of {actual_density:.3}[p/m^2]. (Desired density: {density}[p/m^2].)")

    def _get_pedestrian_id(self, cell_loc=(0, 0)):
        """
        cell_loc -> tuple: (height, width)

        get the ID of the pedestrian at location cell_loc
        """
        p_id = self.state["pedestrian_id"][cell_loc[0], cell_loc[1]]
        assert not p_id == 0
        return p_id

    def _get_pedestrian_index(self, pedestrian_id=1):
        """
        pedestrian_id -> int > 0

        get the index/ location of the pedestrian with the given ID
        """
        assert not pedestrian_id == 0
        p_index = np.hstack(np.nonzero(self.state["pedestrian_id"] == pedestrian_id))
        if p_index.size == 0:
            # no such pedestrian with given id exists
            return None
        # should have only 2 coordinates, if there were more pedestrians with the same id, then this would be larger
        # than 2 (problem: id should be unique)
        assert p_index.shape[0] == 2
        return p_index

    def get_pedestrian_ages(self):
        """
        returns array containing all pedestrian ages, dim: N,
        """
        p_attributes = self.state["pedestrian_attributes"]
        return np.array([p_a["age"] for p_a in p_attributes.values()])

    def get_pedestrian_speeds(self):
        """
        returns array containing all pedestrian ideal speeds, dim: N,
        """
        p_attributes = self.state["pedestrian_attributes"]
        return np.array([p_a["speed"] for p_a in p_attributes.values()])

    def get_pedestrian_sexes_numbers(self):
        """
        returns tuple containing the number of female and male pedestrians, len: 2
        """
        p_attributes = self.state["pedestrian_attributes"]
        sexes = [1 if p_a["sex"] == "male" else 0 for p_a in p_attributes.values()]
        n_male = sum(sexes)
        n_female = len(sexes) - n_male
        return n_female, n_male

    def get_pedestrian_speeds_by_sex(self):
        """
        returns tuple of len=2, containing arrays with the speeds for female and male pedestrians, dim: N, and M,
        """
        speeds_female, speeds_male = [], []
        p_attributes = self.state["pedestrian_attributes"]
        for p_a in p_attributes.values():
            if p_a["sex"] == "male":
                speeds_male.append(p_a["speed"])
            else:
                speeds_female.append(p_a["speed"])
        return speeds_female, speeds_male

    def save_state(self):
        """
        Remembers a state to easily reset to (e.g. the state at the start, before simulation)
        """
        self._compute_constant_cost_map()
        self.saved_state = dict()
        for key, value in self.state.items():
            self.saved_state[key] = value.copy()

    def reset_to_saved_state(self):
        """
        Sets the state back to the one which was saved (if a saved state is available)
        """
        if self.saved_state is None:
            print("No state has been saved.")
        self.state = dict()
        for key, value in self.saved_state.items():
            self.state[key] = value.copy()

    def _compute_constant_cost_map(self, verbose=True):
        """
        verbose -> bool: controls if this prints its progress or not

        computes the constant cost map based on the current state, using the function that was set at instantiation
        """
        if verbose:
            print("Computing constant cost map...")
        self.state = self.constant_cost_map_computation(self.state, self.simulation_parameters.grid_unit)
        if verbose:
            print("Done!")

    def simulate(self, start_at_saved_state, seconds, verbose=True):
        """
        start_at_saved_state -> bool: controls whether the state is reset to the saved one or not before starting
        seconds -> numeric > 0: for how many seconds should the simulation run
        verbose -> bool: controls if this prints its progress or not

        runs the simulation for the given seconds, starting at either the current state, or the last saved one
        """
        # reset the evacuation time
        self.simulation_total_evacuation_time = None
        if verbose:
            print("Simulation is running...")
        steps = int(np.ceil(seconds / self.simulation_parameters.time_step))
        self.simulation_steps = steps
        self.simulation_seconds = steps * self.simulation_parameters.time_step
        if start_at_saved_state:
            # no need to compute the constant cost map -> guaranteed to have been computed before saving the state
            self.reset_to_saved_state()
        else:
            # compute the constant cost map
            self._compute_constant_cost_map(verbose)
        plots = np.zeros((steps + 1, self.grid_size[0], self.grid_size[1]), dtype=np.int8)
        plots[0, :, :] = self._get_plot_data()
        paths = dict()  # {pedestrian_id: path_of_indices}
        p_indices = get_indices_of_state(self.state, "pedestrian")
        for i in range(p_indices.shape[0]):
            p_index = p_indices[i, :]
            p_id = self._get_pedestrian_id(p_index)
            paths[p_id] = np.zeros((2, steps + 1))  # path: 2,T+1
            paths[p_id][:, 0] = p_index
        for i in range(steps):
            if verbose:
                # print the current step as a sort of loading bar
                print(f"\r  time step ({i + 1:>4}  /{steps:>4})", end='', flush=True)
            new_p_locations = self._evolve_one_step(i + 1)
            plots[i + 1, :, :] = self._get_plot_data()
            for p_id, path in paths.items():
                # if the p_id does not exist (pedestrian was absorbed), then repeat its last location
                new_p_index = new_p_locations.get(p_id, path[:, i])
                path[:, i + 1] = new_p_index
        self.simulation_plots = plots
        self.simulation_paths = paths
        if verbose:
            print("\nDone!")

    def _evolve_one_step(self, step):
        """
        step -> int: which is the current step, used to set the total evacuation time

        updates the cells of each pedestrian:
        passes the pedestrian  current state to the evolution operator and sets the new state to
        """
        # get the indices of pedestrians and then permute them randomly -> random order of updates
        p_indices = np.random.permutation(get_indices_of_state(self.state, "pedestrian"))  # N,2
        n_pedestrians = p_indices.shape[0]
        if n_pedestrians == 0 and self.simulation_total_evacuation_time is None:
            # when no pedestrians left: get the evacuation time in seconds
            self.simulation_total_evacuation_time = (step - 1) * self.simulation_parameters.time_step
        new_p_locations = dict()  # {pedestrian_id: index_after_update}
        for i in range(n_pedestrians):
            p_index = p_indices[i, :]
            p_id = self._get_pedestrian_id(p_index)
            self._evolution_operator(p_index, step)
            new_p_location = self._get_pedestrian_index(p_id)
            new_p_locations[p_id] = new_p_location
        return new_p_locations

    def _evolution_operator(self, p_index, step):
        """
        p_index -> tuple: the current pedestrian cell index/ location
        step -> int: which is the current step, used to set the pedestrian evacuation time

        general idea: move the pedestrian to the cell in its neighborhood which has the lowest cost
        """
        p_id = self.state["pedestrian_id"][p_index[0], p_index[1]]
        p_attributes = self.state["pedestrian_attributes"][p_id]
        # to simulate different velocities, we will not move with a probability p (each pedestrian has its own p)
        p = p_attributes["max_speed_fraction"]
        if not move_or_not(p):
            return
        # check if p_index has diag entry above 1 -> as a penalty for too many diagonal moves, stay put -> subtract 1
        if p_attributes["additional_diag_units"] > 1:
            p_attributes["additional_diag_units"] -= 1
            return
        can_move_to_indices = get_movable_neighbor_indices(
            self.state,
            p_index,
            self.grid_size,
            self.absorbing,
            include_obstacles=self.no_obstacle_avoidance,
        )  # N,2
        # the first movable neighbor cell should be the current pedestrian index
        assert np.all(can_move_to_indices[0, :] == p_index)
        if can_move_to_indices.size == 1:
            # no neighbors to move to, only the cell already occupied (stuck) -> don't need the cost computations
            return
        # initialize the cost as the constant cost
        cost = self.state["constant_cost_map"][can_move_to_indices.T[0, :], can_move_to_indices.T[1, :]]  # N,
        # add diagonal movement cost
        cost += get_diagonal_movement_cost(p_index, can_move_to_indices, self.simulation_parameters.grid_unit)
        # for each defined dynamic cost, add it to the total cost
        for dynamic_cost_computation in self.dynamic_cost_computations:
            cost += dynamic_cost_computation(can_move_to_indices, self.state, self.simulation_parameters.grid_unit)
        # which index in cost has the minimum value
        best_i = np.argmin(cost)
        # which neighbor index/ location has the lowest cost
        best_index = can_move_to_indices[best_i, :]
        # check if the cost at best_index is the same as at p_index (current position -> i=0) -> prefer to stay put
        if cost[0] <= cost[best_i]:
            return
        # check if best_index is empty -> we move the pedestrian there
        if self.state["empty"][best_index[0], best_index[1]]:
            self.state = move_pedestrian(self.state, p_index, best_index)
            return
        # check if best_index is a target (and we have absorbing targets) -> target absorbs the pedestrian
        if self.absorbing and self.state["target"][best_index[0], best_index[1]]:
            self.state = absorb_pedestrian(self.state, p_index, best_index)
            # add the evacuation time of this pedestrian to its attributes
            p_attributes["evacuation_time"] = step * self.simulation_parameters.time_step
            return
        if self.no_obstacle_avoidance and self.state["obstacle"][best_index[0], best_index[1]]:
            # this will BREAK the obstacle, and turn it into a pedestrian, but it is intentionally this way, to show
            # how having no obstacle avoidance behaves...
            self.state = move_pedestrian(self.state, p_index, best_index, self.no_obstacle_avoidance)
            return
        raise Exception("Not expected to get to this point in the method, please debug!")

    def _check_if_simulation_data_exists(self):
        """
        raises an error if trying to access data that is only generated during simulation, but no simulation has run yet
        """
        if self.simulation_plots is None:
            raise Exception("Please run the 'simulate' or the 'load_saved_simulation' method first!")

    def get_simulation_evacuation_times_per_pedestrian(self):
        """
        returns dict -> for each pedestrian ID contains the evacuation time
        """
        self._check_if_simulation_data_exists()
        p_attributes = self.state["pedestrian_attributes"]
        return {p_id: p_a["evacuation_time"] for p_id, p_a in p_attributes.items()}

    def _check_simulation_time_interval(self, from_second, to_second):
        """
        from_second -> float: where the time interval begins in simulation time seconds
        to_second -> float: where the time interval ends in simulation time seconds

        returns:
        discrete_from_second -> float: discretized, where the time interval begins in simulation time seconds
        discrete_to_second -> float: discretized, where the time interval ends in simulation time seconds
        int(from_time_step) -> int: at which step the interval starts
        int(to_time_step) -> int: at which step the interval ends

        does all the checks if time interval is valid, and returns the discretized times and steps
        """
        self._check_if_simulation_data_exists()
        step_seconds = self.simulation_parameters.time_step
        max_steps = self.simulation_steps
        max_seconds = self.simulation_seconds
        assert 0 <= from_second < max_seconds
        from_time_step = np.floor(from_second / step_seconds)
        if to_second is None:
            to_time_step = max_steps
        else:
            assert 0 < to_second
            to_time_step = np.ceil(to_second / step_seconds)
        assert from_time_step < to_time_step
        discrete_from_second = from_time_step * step_seconds
        discrete_to_second = to_time_step * step_seconds
        return discrete_from_second, discrete_to_second, int(from_time_step), int(to_time_step)

    def get_simulation_walked_distance_per_pedestrian(self, from_second=0, to_second=None):
        """
        from_second -> float: return distance traveled starting at which second
        to_second-> float or None: return distance traveled until which second, if None, until end

        returns:
        dicts -> for every pedestrian ID: the distance traveled
        """
        from_s, _, from_time_step, to_time_step = self._check_simulation_time_interval(from_second, to_second)
        p_attributes = self.state["pedestrian_attributes"]
        evacuation_seconds = {p_id: p_att["evacuation_time"] for p_id, p_att in p_attributes.items()}
        distance_per_pedestrian = dict()
        for p_id, path in self.simulation_paths.items():
            evacuation_s = evacuation_seconds[p_id]
            if evacuation_s is not None and evacuation_s <= from_s:
                # pedestrian was already evacuated before the start of time interval
                distance_per_pedestrian[p_id] = None
                continue
            time_step_distances = np.sqrt(np.sum(np.diff(path[:, from_time_step:to_time_step+1]) ** 2, axis=0))
            # convert to meters
            distance_per_pedestrian[p_id] = np.sum(time_step_distances) * self.simulation_parameters.grid_unit
        return distance_per_pedestrian

    def get_simulation_average_walked_distance(self, from_second=0, to_second=None):
        """
        from_second -> float: return distance traveled starting at which second
        to_second-> float or None: return distance traveled until which second, if None, until end

        returns:
        float -> the average distance traveled by all pedestrians during given time interval
        """
        walked_distance_per_pedestrian = self.get_simulation_walked_distance_per_pedestrian(from_second, to_second)
        distances = [d for d in walked_distance_per_pedestrian.values() if d is not None]
        return np.mean(distances)

    def get_simulation_average_speed_per_pedestrian(self, from_second=0, to_second=None):
        """
        from_second -> float: return distance traveled starting at which second
        to_second-> float or None: return distance traveled until which second, if None, until end

        returns:
        dicts -> for every pedestrian ID: the average speed at which they walked during time interval
        """
        from_s, to_s, _, _ = self._check_simulation_time_interval(from_second, to_second)
        distances_walked = self.get_simulation_walked_distance_per_pedestrian(from_second, to_second)
        p_attributes = self.state["pedestrian_attributes"]
        evacuation_seconds = {p_id: p_att["evacuation_time"] for p_id, p_att in p_attributes.items()}
        time_walked = to_s - from_s
        avg_speeds = dict()
        for p_id, distance_walked in distances_walked.items():
            if distance_walked is None:
                # pedestrian was absorbed before from_seconds
                avg_speeds[p_id] = None
                continue
            evacuation_s = evacuation_seconds[p_id]
            if evacuation_s is not None and evacuation_s < to_s:
                # means the pedestrian was evacuated before the to_time_step (end) was reached
                time_walked_to_evacuation = evacuation_s - from_s
                assert 0 < time_walked_to_evacuation
                avg_speeds[p_id] = distance_walked / time_walked_to_evacuation
            else:
                # walked the entire time
                avg_speeds[p_id] = distance_walked / time_walked
        return avg_speeds

    def get_simulation_average_speed(self, from_second=0, to_second=None):
        """
        from_second -> float: return distance traveled starting at which second
        to_second-> float or None: return distance traveled until which second, if None, until end

        returns:
        float -> the average speed of all pedestrians during given time interval
        """
        avg_speed_per_pedestrian = self.get_simulation_average_speed_per_pedestrian(from_second, to_second)
        speeds = [s for s in avg_speed_per_pedestrian.values() if s is not None]
        return np.mean(speeds)

    def _get_plot_data(self):
        """
        return: numpy array with same dimensions as grid size, containing the values to be plotted, dim: N,M

        transforms the 4 boolean matrices into a single one with different values for each state variable
        """
        plot_states = ["empty", "pedestrian", "obstacle", "target"]
        state_tensor = np.zeros((len(plot_states), self.grid_size[0], self.grid_size[1]), dtype=np.int8)
        for i, key in enumerate(plot_states):
            state_tensor[i, :, :] = self.state[key]
        return np.sum(state_tensor * np.arange(len(plot_states))[:, None, None], axis=0)

    def _plot_data(self, plot_data, ax):
        """
        plot_data -> numpy array: as the one produced by self._get_plot_data()
        ax -> a matplotlib axes object, will plot onto given axes

        returns: matplotlib image object, can be used later to update the data in interactive plot

        plots the given matrix as an image with different colors for each state variable
        """
        im = ax.imshow(plot_data,
                       cmap=self.plot_color_map,
                       extent=(0, self.grid_size[1], self.grid_size[0], 0),
                       vmin=0, vmax=3, zorder=2, alpha=get_plot_alphas(plot_data))
        add_meter_units_to_plot(ax, self.grid_size, self.simulation_parameters.grid_unit)
        ax.grid(visible=False)
        return im

    def plot_state(self, ax=None, fig_size=None):
        """
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the current state with the dimensions
        """
        ax = get_plot_ax(ax, fig_size)
        _ = self._plot_data(self._get_plot_data(), ax)
        plt.show()

    def plot_constant_cost_map(self, ax=None, fig_size=None, recompute=False):
        """
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)
        recompute -> bool: determines whether the cost map is recomputed before plotting

        plots the constant cost map -> only computed before the simulation
        """
        if recompute:
            self._compute_constant_cost_map()
        ax = get_plot_ax(ax, fig_size)
        im = ax.imshow(self.state["constant_cost_map"], cmap=self.cost_color_map,
                       extent=(0, self.grid_size[1], self.grid_size[0], 0))
        cbar = plt.colorbar(im)
        cbar.set_label("constant cell cost")
        add_meter_units_to_plot(ax, self.grid_size, self.simulation_parameters.grid_unit)
        ax.grid(visible=False)

    def plot_simulation_end_state(self, ax=None, fig_size=None):
        """
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plot the end state of the simulation
        """
        self._check_if_simulation_data_exists()
        ax = get_plot_ax(ax, fig_size)
        end_plot = self.simulation_plots[-1, :, :]
        _ = plot_paths(self.simulation_paths, ax, steps=self.simulation_steps)
        _ = self._plot_data(end_plot, ax)
        plt.show()

    def plot_simulation_with_time_slider(self, fig_size=None):
        """
        fig_size -> tuple: controls the plot figure size, (width, height)

        displays the simulation run as an interactive plot with a time slider
        For the slider to remain responsive reference to it must be maintained, that is why we save to object
        """
        self._check_if_simulation_data_exists()
        fig = plt.figure()
        if fig_size is not None:
            fig.set_size_inches(fig_size)
        ax_plots = fig.add_axes([0.1, 0.13, 0.8, 0.77])
        current_height_inches = fig.get_size_inches()[1]
        slider_height = 0.03 * (5 / current_height_inches)  # 3% of 5 inch figure is a good slider height
        ax_step_slider = fig.add_axes([0.15, 0.05, 0.7, slider_height])
        lines = plot_paths(self.simulation_paths, ax_plots, steps=0)
        im = self._plot_data(self.simulation_plots[0, :, :], ax_plots)
        step_slider = create_step_slider(
            fig,
            im,
            ax_step_slider,
            self.simulation_plots,
            self.simulation_paths,
            lines,
            self.simulation_parameters.time_step,
        )
        self.step_sliders.append(step_slider)
        plt.show()

    def plot_simulation_animation(self, speed_up=1.0, fig_size=None):
        """
        speed_up -> float: controls how much faster/ slower the animation should run compared to the simulation time
        fig_size -> tuple: controls the plot figure size, (width, height)

        displays the simulation as an animation
        """
        self._check_if_simulation_data_exists()
        fig, ax = plt.subplots()
        if fig_size is not None:
            fig.set_size_inches(fig_size)
        lines = plot_paths(self.simulation_paths, ax, steps=0)
        im = self._plot_data(self.simulation_plots[0, :, :], ax)
        steps = self.simulation_plots.shape[0]
        path_data = [{p_id: p_total_path[:, :i+1] + 0.5
                      for p_id, p_total_path in self.simulation_paths.items()}
                     for i in range(steps)]
        plot_data = [(self.simulation_plots[i, :, :], get_plot_alphas(self.simulation_plots[i, :, :]))
                     for i in range(steps)]
        frame_data = list(zip(path_data, plot_data))

        def update_animation(current_frame_data):
            line_data_dict, frame_plot_data = current_frame_data
            map_data, alpha_data = frame_plot_data
            artists = []
            for p_id, line_data in line_data_dict.items():
                lines[p_id].set_data(line_data[1, :], line_data[0, :])
                artists.append(lines[p_id])
            im.set(data=map_data, alpha=alpha_data)
            artists.append(im)
            return artists

        interval = (self.simulation_parameters.time_step * 1000) / speed_up  # ms
        ca_animation = FuncAnimation(
            fig, update_animation, frames=frame_data, interval=interval, repeat_delay=1000, blit=True
        )
        return ca_animation

    def video_simulation_animation(self, speed_up=1.0, fig_size=None):
        """
        speed_up -> float: controls how much faster/ slower the animation should run compared to the simulation time
        fig_size -> tuple: controls the plot figure size, (width, height)

        returns: str: encoded HTML5 video, can be displayed in jupyter notebook

        takes quite some time to compute, requires an ffmpeg installation
        """
        ca_animation = self.plot_simulation_animation(speed_up, fig_size)
        html5_video_animation = ca_animation.to_html5_video()
        plt.close()
        return html5_video_animation

    def plot_simulation_average_speed_per_time_interval(self, intervals=10, ax=None, fig_size=None):
        """
        intervals -> int: divide the simulation time in how many intervals
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the average speed during each time interval
        """
        ax = get_plot_ax(ax, fig_size)
        interval_edges = np.linspace(0, self.simulation_seconds, intervals)
        interval_size = interval_edges[1] - interval_edges[0]
        interval_centers = interval_edges[:-1] + interval_size / 2
        avg_speeds = []
        for i in range(interval_edges.size - 1):
            avg_speeds.append(self.get_simulation_average_speed(interval_edges[i], interval_edges[i+1]))
        ax.bar(interval_centers, avg_speeds, width=interval_size, alpha=0.5, ec="k",
               label="Average pedestrian speed in time interval")
        y_lim = ax.get_ylim()
        ax.set_ylim(y_lim[0], y_lim[1] * 1.2)
        ax.set_xlabel("simulation time [s]")
        ax.set_ylabel("speed [m/s]")
        ax.legend()

    def plot_simulation_speed_function_of_age(self, n_bins=5, ax=None, fig_size=None):
        """
        n_bins -> int: divide age data into how many bins
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the speed-age function reconstructed from current pedestrians actual simulation speed on top of the
        actual function
        """
        self._check_if_simulation_data_exists()
        ax = get_plot_ax(ax, fig_size)
        p_attributes = self.state["pedestrian_attributes"]
        speed_per_pedestrian = self.get_simulation_average_speed_per_pedestrian()
        speeds = []
        ages = []
        for p_id, speed in speed_per_pedestrian.items():
            speeds.append(speed)
            ages.append(p_attributes[p_id]["age"])
        ages = np.array(ages)
        speeds = np.array(speeds)
        self.simulation_parameters.plot_speed_function_from_samples(ax, ages, speeds, n_bins=n_bins)

    def plot_age_histogram(self, n_bins=5, ax=None, fig_size=None):
        """
        n_bins -> int: divide age data into how many bins
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the age histogram from current pedestrians on top of the actual distribution
        """
        ax = get_plot_ax(ax, fig_size)
        ages = self.get_pedestrian_ages()
        self.simulation_parameters.plot_age_histogram(ax=ax, ages=ages, n_bins=n_bins)

    def plot_sexes_histogram(self, ax=None, fig_size=None):
        """
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the sexes histogram from current pedestrians together with the expected fraction
        """
        ax = get_plot_ax(ax, fig_size)
        n_female, n_male = self.get_pedestrian_sexes_numbers()
        self.simulation_parameters.plot_sexes_frequency_histogram(ax, n_female, n_male)

    def plot_sexes_mean_speed(self, ax=None, fig_size=None):
        """
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the sexes mean speeds from current pedestrians together with the expected means
        """
        ax = get_plot_ax(ax, fig_size)
        speeds_female, speeds_male = self.get_pedestrian_speeds_by_sex()
        self.simulation_parameters.plot_sexes_mean_speed(ax, speeds_female, speeds_male)

    def plot_speed_function_of_age(self, n_bins=5, ax=None, fig_size=None):
        """
        n_bins -> int: divide age data into how many bins
        ax -> matplotlib axes: if given plots onto ax
        fig_size -> tuple: controls the plot figure size, (width, height)

        plots the speed-age function reconstructed from current pedestrians ideal speed on top of the actual function
        """
        ax = get_plot_ax(ax, fig_size)
        ages = self.get_pedestrian_ages()
        speeds = self.get_pedestrian_speeds()
        self.simulation_parameters.plot_speed_function_from_samples(ax, ages, speeds, n_bins=n_bins)


def get_plot_ax(ax=None, fig_size=None):
    """
    ax -> matplotlib axes: if None, creates one
    fig_size -> tuple: controls the plot figure size, (width, height)

    returns: matplotlib axes, of the figure with adjusted fig_size
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if fig_size is not None:
        fig.set_size_inches(fig_size)
    return ax
