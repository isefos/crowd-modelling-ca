import numpy as np
import matplotlib.pyplot as plt


def get_equally_spaced_circle_rad_angles(number_of_angles):
    """
    number_of_angles -> int

    return -> numpy array of dim number_of_angles,
    """
    two_pi = 2 * np.pi
    # random angle between -45° and 45°
    start_angle = (np.random.uniform() - 0.5) * two_pi / 8
    return start_angle + np.linspace(0, two_pi, number_of_angles, endpoint=False)


def get_n_equally_spaced_points_on_grid_circle(n, center, radius):
    """
    n -> int
    center -> tuple: len=2, int 2D coordinates
    radius -> float

    return -> numpy array dim n,
    """
    angles = get_equally_spaced_circle_rad_angles(n)
    points_on_unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    points_on_zero_center_circle = points_on_unit_circle * radius
    points_on_circle = points_on_zero_center_circle + np.array(center)[:, None]
    return np.around(points_on_circle).astype(int).T


def plot_average_velocities_per_time_interval(ca_dict):
    """
    ca_dict -> dict: CA models

    Used to plot various different model plots onto the same figure
    """
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches((10, 6))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.17, wspace=0.5, hspace=0.5)
    for i, (density, ca) in enumerate(ca_dict.items()):
        ax = axs.flatten()[i]
        ca.plot_simulation_average_speed_per_time_interval(ax=ax)
        ax.set_title(f"Density {density}" + r"$\left[\frac{p}{m^2}\right]$")
        ax.get_legend().remove()
    axs.flatten()[-1].set_axis_off()


def plot_fundamental_diagrams(densities, speeds):
    """
    densities, speeds -> lists of same length: float
    """
    flow = np.array(densities) * np.array(speeds)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches((10, 4))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.17, wspace=0.3)
    density_label = r"pedestrian density $\left[\frac{p}{m^2}\right]$"
    speed_label = r"average pedestrian speed $\left[\frac{m}{s}\right]$"
    flow_label = r"average pedestrian flow $\left[\frac{p}{m*s}\right]$"
    _plot_part_fundamental_diagram(axs[0], densities, speeds, density_label, speed_label, "Speed per density")
    _plot_part_fundamental_diagram(axs[1], speeds, flow, speed_label, flow_label, "Flow per speed", x_scale=1.1)
    _plot_part_fundamental_diagram(axs[2], densities, flow, density_label, flow_label, "Flow per density")


def _plot_part_fundamental_diagram(ax, x_argument, y_argument, x_label, y_label, title, x_scale=1.5):
    """
    ax -> matplotlib axes
    x_argument -> iterable of numerics
    y_argument -> iterable of numerics of same length
    x_label -> str
    y_label -> str
    title -> str
    x_scale -> float
    """
    ax.plot(x_argument, y_argument, "k", alpha=0.8)
    ax.set_xlabel(x_label)
    x_lim = ax.get_xlim()
    ax.set_xlim(0, x_lim[1] * x_scale)
    ax.set_ylabel(y_label)
    y_lim = ax.get_ylim()
    ax.set_ylim(0, y_lim[1] * 1.1)
    ax.set_title(title)
    ax.grid()
