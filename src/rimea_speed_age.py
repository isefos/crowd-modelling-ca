"""
In this file we simply try to approximate speed age function in RiMEA figure 2 with a piecewise linear one
"""


def get_line_parameters_from_2_points(point_1, point_2):
    """
    point_1 -> tuple len=2: float 2D coordinates
    point_2 -> tuple len=2: float: 2D coordinates

    return ->  tuple len=2: float

    get the linear equation parameters from two points

    points -> (x, y)
    line -> y = (m * x) + b
    line parameters -> m = (y1 - y2) / (x1 - x2) -> b = y1 - (m * x1)
    """
    coefficient = ((point_1[1] - point_2[1]) / (point_1[0] - point_2[0]))
    bias = point_1[1] - coefficient * point_1[0]
    return coefficient, bias


def get_population_speed_distribution_parameters(age_values, mean_points, top_std_points):
    """
    age_values, mean_points, top_std_points -> list of floats and equal length

    return -> tuple, len=2: dict

    defined points taken from the graph in figure 2 in RiMEA. used to output the piecewise linear approximation
    equations for the mean and standard deviation.

    linear equations:
    mean = m_mean_i * age + b_mean_i, if age in input range i
    std = m_std_i * age + b_std_i, if age in input range i
    """
    n_points = len(age_values)
    assert len(mean_points) == len(top_std_points) == n_points
    ms = {"mean": [], "std": []}
    bs = {"mean": [], "std": []}
    for i in range(n_points - 1):
        age_p1, age_p2 = age_values[i:i+2]
        mean_p1, mean_p2 = mean_points[i:i+2]
        std_p1, std_p2 = top_std_points[i:i+2]
        m_mean, b_mean = get_line_parameters_from_2_points((age_p1, mean_p1), (age_p2, mean_p2))
        ms["mean"].append(m_mean)
        bs["mean"].append(b_mean)
        biased_m_std, biased_b_std = get_line_parameters_from_2_points((age_p1, std_p1), (age_p2, std_p2))
        m_std = biased_m_std - m_mean
        b_std = biased_b_std - b_mean
        ms["std"].append(m_std)
        bs["std"].append(b_std)
    return ms, bs


rimea_ages = [3, 10, 20, 30, 40, 50, 60, 70, 80, 85]
rimea_means = [0.56, 1.17, 1.6, 1.53, 1.48, 1.41, 1.26, 1.06, 0.7, 0.55]
rimea_top_stds = [0.66, 1.4, 1.91, 1.81, 1.76, 1.65, 1.5, 1.26, 0.8, 0.59]
# compute the coefficient and biases for above defined RiMEA data
ms_rimea, bs_rimea = get_population_speed_distribution_parameters(rimea_ages, rimea_means, rimea_top_stds)
