"""
In this file we are mostly concerned with getting the correct population and physical attributes for the simulation.
Most of the data are based on the RiMEA guidelines.
"""
from src.rimea_speed_age import (
    rimea_ages,
    ms_rimea,
    bs_rimea,
    get_population_speed_distribution_parameters,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, truncnorm


def get_truncated_normal_random_variable(mean, std, trunc_lower, trunc_upper):
    return truncnorm((trunc_lower - mean) / std, (trunc_upper - mean) / std, loc=mean, scale=std)


class SimulationParameters:
    def __init__(
            self,
            grid_unit=0.4,  # m
            max_speed=2.5,  # m/s
            females_per_male=0.5,  # in interval [0, 1]
            age_mean=50,
            age_std=20,
            age_trunc_lower=10,  # minimum possible age
            age_trunc_upper=85,  # maximum possible age
            speed_function_age_values=rimea_ages,  # ages for points from the speed (mean and std) by age function
            speed_function_coefficients=ms_rimea,  # dict: coefficients for mean and std lines between given age points
            speed_function_biases=bs_rimea,  # dict: biases for mean and std lines between given age points
            male_by_female_speed_factor=1.109,  # mean male speed / mean female speed
            desired_mean_male_speed=1.41,  # m/s
            desired_mean_female_speed=1.27,  # m/s
    ):
        """
        ----------------------------------------------------------------------------------------------------------------
        Physical dimensions (defaults):

        Each cell has the dimensions 40cm x 40cm and can be occupied by a single pedestrian.
        Since in the cellular automaton we only allow for movement to a neighboring cell (inside the 9x9 grid, with the
        current cell at the center), if a pedestrian moves in the same direction, one cell at every time step, its
        velocity is: 0.4 m / u s , where u is the unit of time of a single time step. This corresponds to the maximal
        velocity.
        Based on available data from RiMEA, we want the top speed to be 2.5 m/s. Therefore, we can solve for the time
        unit of a single time step: 0.4 m / u s = 2.5 m/s, u = 0.4 m / (2.5 m/s) = 0.16 s . As a result, each update
        step of the cellular automaton can be interpreted as a discrete time jump of 0.16 seconds.
        ----------------------------------------------------------------------------------------------------------------
        Sex (defaults):

        p = N_females / N_males
        From RiMEA: 50 % male, 50% female -> bernoulli distribution with p = 0.5
        ----------------------------------------------------------------------------------------------------------------
        Age (defaults):

        implement the distribution given in figure 3 of RiMEA: a truncated normal distribution with:
        range limits: from age 10 to 85 (truncation)
        mean = 50
        std = 20
        ----------------------------------------------------------------------------------------------------------------
        Speed (defaults):

        pedestrian mean speed and speed standard deviation is given as a function of age in RiMEA figure 2
        additionally, male pedestrians should be on average 10,9 % faster than female ones
        on average, male pedestrians should have a speed of 1.41 m/s, and female pedestrians 1.27 m/s.
        we can try to reconstruct the function from figure 2 with a piecewise linear function,
        then we can get the mean and standard deviation of a given age, and sample from a normal distribution with those
        parameters.
        the standard deviation seems to be mostly symmetric (and in the legend it says mean +- sigma, which suggests
        that sigma is the same) -> so only reconstructing one of the lines should be sufficient
        unfortunately, in the figure, only the speeds for ages up to 80 years are shown, in our population however, it
        is supposed to be possible to have pedestrians of an age up to 85. Therefore, we will just try to continue the
        trend line of the rest of the graph for 80 - 85 (based purely on how good it looks visually, and hopefully we
        won't sample many pedestrians in that age group anyway).

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Speed distribution for the different sexes (defaults):

        Given the age, get the corresponding mean and standard deviation of the total speed distribution. Then correct
        these parameters for the sex of the pedestrian:
          -> "male pedestrians should be on average 10,9 % faster than female ones"
        So we have two different distributions at each given age, one for males and one for females. Assumptions:
         - The joint distribution should result in the given total mean, given by the speed function at a given age
         - The joint distribution should have the given total variance, given by the speed function at a given age
         - The variances for the female and male distributions are the same (our assumption)
         - the total population has a factor p = N_female / N_male (0.5 by default)
        Using these assumptions we can find the scaling factors to apply to the total parameters to get the individual
        sex ones. Then we sample from a truncated normal distribution with the corresponding mean and std (truncated
        between 0 and max_speed).
        ----------------------------------------------------------------------------------------------------------------
        """
        self.grid_unit = grid_unit  # m
        self.max_speed = max_speed  # m/s
        self.time_step = grid_unit / max_speed
        # for the sex distribution:
        p = females_per_male
        self.females_per_male = p
        self.pedestrian_sex_random_variable = bernoulli(p)
        # for the age distribution:
        self.age_min = age_trunc_lower
        self.age_max = age_trunc_upper
        self.age_mean = age_mean
        self.age_std = age_std
        self.age_random_variable = get_truncated_normal_random_variable(mean=age_mean,
                                                                        std=age_std,
                                                                        trunc_lower=age_trunc_lower,
                                                                        trunc_upper=age_trunc_upper)
        # for the function that specifies the speed distribution parameters by a given age:
        # check if we have fully defined a speed for every possible age
        assert min(speed_function_age_values) <= self.age_min
        assert max(speed_function_age_values) >= self.age_max
        self.sf_age_values = speed_function_age_values
        self.sf_ms = speed_function_coefficients
        self.sf_bs = speed_function_biases
        # for the speed distributions by sex
        f = male_by_female_speed_factor
        self.male_by_female_speed_factor = f
        self.desired_mean_male_speed = desired_mean_male_speed
        self.desired_mean_female_speed = desired_mean_female_speed
        # check if the factor and the desired speeds somewhat align (allow up to 3% difference)
        assert ((desired_mean_male_speed / desired_mean_female_speed) - f <= 0.03)
        a = 1 / (p + f - p * f)
        self.female_by_total_speed_factor = a
        b = f * a
        self.male_by_total_speed_factor = b
        self.constant_std_scaling_part = p * (1 - p) * (a - b) ** 2

    def get_sex_from_distribution(self):
        """
        return -> str

        random sample from the pedestrian sex random variable
        """
        binary = self.pedestrian_sex_random_variable.rvs()
        if binary:
            return "female"
        else:
            return "male"

    def get_age_from_distribution(self):
        """
        return -> float

        random sample from the age random variable
        """
        return self.age_random_variable.rvs()

    def plot_age_distribution_pdf(self, ax=None):
        """
        ax -> matplotlib axes

        return -> matplotlib axes

        plots the age distribution pdf onto a given axis, or creates a new figure if no axis is given
        returns the axis
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ages = np.linspace(self.age_min, self.age_max, 200)
        ax.plot(ages, self.age_random_variable.pdf(ages), "k", label="age pdf")
        ax.set_xlabel("age")
        ax.set_xlim(left=0, right=self.age_max + 5)
        ax.set_xticks(np.arange(start=0, stop=self.age_max + 10, step=5))
        ax.set_ylabel("age probability density function")
        ax.set_ylim(bottom=0)  # top=0.025
        ax.grid(visible=True, axis="both")
        return ax

    def get_speed_mean_and_std(self, given_age):
        """
        given_age -> float

        return -> tuple len=2: float

        returns the mean speed and standard deviation, given a certain age.
        """
        for i in range(len(self.sf_age_values) - 1):
            age_1, age_2 = self.sf_age_values[i:i+2]
            if age_1 <= given_age <= age_2:
                speed_mean = self.sf_ms["mean"][i] * given_age + self.sf_bs["mean"][i]
                speed_std = self.sf_ms["std"][i] * given_age + self.sf_bs["std"][i]
                return speed_mean, speed_std
        raise Exception(f"given_age = {given_age} is not in the valid range: "
                        f"[{self.sf_age_values[0]}-{self.sf_age_values[-1]}]!]")

    def plot_speed_function(self, ax=None):
        """
        ax -> matplotlib axes

        return -> matplotlib axes

        plots the piecewise linear approximation of the speed mean and standard deviation, given the age.
        """
        if ax is None:
            fig, ax = plt.subplots()
        min_sf_age = self.sf_age_values[0]
        max_sf_age = self.sf_age_values[-1]
        ages = np.linspace(min_sf_age, max_sf_age, 200)
        mean, top_std, bottom_std = get_speed_mean_and_std_array(ages, self.sf_age_values, self.sf_ms, self.sf_bs)
        ax.plot(ages, mean, "k", label=r"mean walking speed $\mu$ [m/s]")
        ax.plot(ages, top_std, "k", label=r"$\mu + \sigma$ [m/s]", linestyle=":")
        ax.plot(ages, bottom_std, "k", label=r"$\mu - \sigma$ [m/s]", linestyle="--")
        ax.set_xlabel("age [years]")
        ax.set_xlim(left=0, right=max_sf_age+5)
        ax.set_xticks(np.arange(start=0, stop=max_sf_age+10, step=10))
        ax.set_ylabel("horizontal walking speed [m/s]")
        ax.set_ylim(bottom=0, top=self.max_speed)
        ax.grid(visible=True, axis="both")
        ax.legend()
        return ax

    def get_speed_from_distribution(self, age, sex):
        """
        age -> float
        sex -> str: either 'female' or 'male'

        return -> float

        gets a speed value from the random distribution corresponding to the given age and sex
        """
        mean, std = self.get_speed_mean_and_std(age)
        # how much we scale the standard deviation depends on the total standard deviation and mean we want to achieve
        std_scaling = np.sqrt(1 - self.constant_std_scaling_part * (mean ** 2) / (std ** 2))
        std *= std_scaling
        if sex == "male":
            mean *= self.male_by_total_speed_factor
        elif sex == "female":
            mean *= self.female_by_total_speed_factor
        else:
            raise Exception(f"Invalid sex value {sex}, only 'male' and 'female' supported in the simulation model!")
        speed = get_truncated_normal_random_variable(mean, std, 0, self.max_speed).rvs()  # in m/s
        return speed

    def get_max_speed_fraction(self, speed):
        """
        speed -> float

        return -> float between 0 and 1
        """
        fraction_of_max = speed / self.max_speed
        assert 0 <= fraction_of_max <= 1
        return fraction_of_max

    def get_speed_from_fraction(self, max_speed_fraction):
        """
        max_speed_fraction -> float between 0 and 1

        return -> float
        """
        assert 0 <= max_speed_fraction <= 1
        return self.max_speed * max_speed_fraction

    def test_distribution_sampling(self, n_samples=1000):
        """
        n_samples -> int

        Checks if we get the desired global statistics when sampling.
        """
        print("Sampling...")
        ages = np.zeros(n_samples)
        sexes = np.zeros(n_samples, dtype=np.int8)  # 1 = female, 0 = male
        speeds = np.zeros(n_samples)
        for i in range(n_samples):
            age = self.get_age_from_distribution()
            ages[i] = age
            sex = self.get_sex_from_distribution()
            if sex == "female":
                sexes[i] = 1
            speeds[i] = self.get_speed_from_distribution(age, sex)
        print("Done!\n\n")
        mean_age = np.mean(ages)
        std_age = np.std(ages)
        p = self.females_per_male
        n_female = np.sum(sexes)
        p_sampled = n_female / n_samples
        mean_speed = np.mean(speeds)
        speeds_female = speeds[sexes == 1]
        mean_female_speed = np.mean(speeds_female)
        speeds_male = speeds[sexes == 0]
        mean_male_speed = np.mean(speeds_male)
        f = self.male_by_female_speed_factor
        f_sampled = mean_male_speed / mean_female_speed
        print(f"Sampled mean age: {mean_age:.3}\n  ---  Desired: {float(self.age_mean):.3}\n"
              f"Sampled age std: {std_age:.3}\n  ---  Desired: {float(self.age_std):.3}\n"
              f"Sampled proportion females / males: {p_sampled:.3}\n  ---  Desired: {self.females_per_male:.3}\n"
              f"Sampled mean speed increase males vs females: {100 * (f_sampled - 1):.3}%\n  ---  "
              f"Desired: {100 * (f - 1):.3}%\n"
              f"Sampled mean speed females: {mean_female_speed:.3}\n  ---  "
              f"Desired: {self.desired_mean_female_speed:.3}\n"
              f"Sampled mean speed males: {mean_male_speed:.3}\n  ---  Desired: {self.desired_mean_male_speed:.3}\n"
              f"Sampled mean speed: {mean_speed:.3}\n  ---  "
              f"Desired: {p * self.desired_mean_female_speed + (1 - p) * self.desired_mean_male_speed:.3}\n")
        # plots to show distribution
        fig = plt.figure()
        # plot 1: age histogram on top of distribution
        ax1 = fig.add_subplot(3, 1, 1)
        self.plot_age_histogram(ax1, ages)
        # plot 2: females vs males bar plot
        ax2 = fig.add_subplot(3, 2, 3)
        self.plot_sexes_frequency_histogram(ax2, n_female, n_samples - n_female)
        # plot 3: mean speeds per sex
        ax3 = fig.add_subplot(3, 2, 4)
        self.plot_sexes_mean_speed(ax3, speeds_female, speeds_male)
        # plot 4: mean and std at each age histogram bin on top of the speed function
        ax4 = fig.add_subplot(3, 1, 3)
        self.plot_speed_function_from_samples(ax4, ages, speeds)
        fig_width, fig_height = fig.get_size_inches()
        fig.set_figheight(3 * fig_height)
        fig.subplots_adjust(left=0.12, right=0.92, top=0.98, bottom=0.04, wspace=0.3)

    def plot_age_histogram(self, ax, ages, n_bins=20):
        """
        ax -> matplotlib axes
        ages -> iterable of floats
        n_bins -> int

        plots a histogram of ages on top of the correct pdf of ages
        """
        if ax is None:
            fig, ax = plt.subplots()
        hist, bin_edges, bin_centers, width = get_histogram_bins(ages, n_bins)
        normalized_hist = hist / np.sum(hist * width)
        _ = self.plot_age_distribution_pdf(ax)
        ax.bar(bin_centers, normalized_hist, width=width, label="sampled relative frequency", alpha=0.5, ec="k")
        _, y_top_limit = ax.get_ylim()
        histogram_y_limit = 1.15 * np.max(normalized_hist)
        if histogram_y_limit * 1.2 > y_top_limit:
            ax.set_ylim(0, histogram_y_limit)
        ax.legend()

    def plot_sexes_frequency_histogram(self, ax, n_female, n_male):
        """
        ax -> matplotlib axes
        n_female -> int
        n_male -> int
        """
        if ax is None:
            fig, ax = plt.subplots()
        relative_frequency = np.array([n_female, n_male]) / (n_female + n_male)
        expected_relative_frequency = [self.females_per_male, 1 - self.females_per_male]
        plot_sex_histogram(ax, relative_frequency, expected_relative_frequency, "sampled relative frequency")

    def plot_sexes_mean_speed(self, ax, speeds_female, speeds_male):
        """
        ax -> matplotlib axes
        speeds_female -> iterable of floats
        speeds_male -> iterable of floats
        """
        if ax is None:
            fig, ax = plt.subplots()
        mean_speeds = [np.mean(speeds_female), np.mean(speeds_male)]
        std_speeds = [np.std(speeds_female), np.std(speeds_male)]
        expected_mean_speeds = [self.desired_mean_female_speed, self.desired_mean_male_speed]
        plot_sex_histogram(ax, mean_speeds, expected_mean_speeds, "sampled mean speed [m/s]", std_speeds)

    def plot_speed_function_from_samples(self, ax, ages, speeds, n_bins=20):
        """
        ax -> matplotlib axes
        ages -> iterable of floats
        speeds -> iterable of floats
        n_bins -> int
        """
        if ax is None:
            fig, ax = plt.subplots()
        _, bin_edges, bin_centers, _ = get_histogram_bins(ages, n_bins)
        _ = self.plot_speed_function(ax)
        bin_indices = np.digitize(ages, bin_edges)
        # put the rightmost limit into the last bin (instead of being a single outsider)
        if np.max(bin_indices) > bin_edges.size:
            bin_indices[np.argmax(bin_indices)] = bin_edges.size
        # set the bin indices to start at zero
        bin_indices -= 1
        speed_means = np.zeros_like(bin_centers)
        speed_stds = np.zeros_like(bin_centers)
        for i in range(bin_centers.size):
            bin_mask = bin_indices == i
            if bin_mask.sum() <= 2:
                raise Exception("Please reduce n_bins, such that all bins have at least two points!")
            speed_means[i] = np.mean(speeds[bin_mask])
            speed_stds[i] = np.std(speeds[bin_mask])
        top_speed_stds = speed_means + speed_stds
        ms, bs = get_population_speed_distribution_parameters(bin_centers, speed_means, top_speed_stds)
        ages = np.linspace(bin_centers[0], bin_centers[-1], 200)
        mean, top_std, bottom_std = get_speed_mean_and_std_array(ages, bin_centers, ms, bs)
        ax.plot(ages, mean, "r", label=r"$\mu_{sample}$ [m/s]")
        ax.plot(ages, top_std, "r", label=r"$\mu_{sample} + \sigma_{sample}$ [m/s]", linestyle=":")
        ax.plot(ages, bottom_std, "r", label=r"$\mu_{sample} - \sigma_{sample}$ [m/s]", linestyle="--")
        ax.legend(ncol=2)


def get_histogram_bins(data, n_bins):
    """
    data -> iterable of numerics
    n_bins -> int

    return -> tuple len=4, (numpy array, numpy array, numpy array, float)
    """
    hist, bin_edges = np.histogram(data, n_bins)
    width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + width / 2
    return hist, bin_edges, bin_centers, width


def get_speed_mean_and_std_array(given_ages, sf_age_values, sf_ms, sf_bs):
    """
    given_ages -> numpy array, dim N
    sf_age_values -> numpy array, dim M
    sf_ms -> dict of coefficients
    sf_bs -> dict of biases

    return -> tuple len=4, (numpy array dim N, numpy array dim N, numpy array dim N)

    calculates the speed mean and standard deviation for an entire array of given ages.
    """
    speed_mean = np.zeros_like(given_ages)
    speed_top_std = np.zeros_like(given_ages)
    speed_bottom_std = np.zeros_like(given_ages)
    for i in range(len(sf_age_values) - 1):
        age_1, age_2 = sf_age_values[i:i+2]
        ages_mask = (age_1 <= given_ages) * (given_ages <= age_2)
        speed_mean[ages_mask] = (sf_ms["mean"][i] * given_ages[ages_mask] + sf_bs["mean"][i])
        speed_top_std[ages_mask] = speed_mean[ages_mask] + (sf_ms["std"][i] * given_ages[ages_mask] + sf_bs["std"][i])
        speed_bottom_std[ages_mask] = speed_mean[ages_mask] - (sf_ms["std"][i] * given_ages[ages_mask] + sf_bs["std"][i])
    return speed_mean, speed_top_std, speed_bottom_std


def plot_sex_histogram(ax, values, expected_values, axis_label, errors=None):
    """
    ax -> matplotlib axes
    values -> tuple len=2: numeric
    expected_values -> tuple len=2: numeric
    axis_label -> str
    errors -> tuple len=2: numeric

    order: index 0 -> female, index 1 -> male
    """
    ax.plot([0.05, 0.95], [expected_values[0]]*2, "r", alpha=0.75, label="expectation female")
    ax.plot([1.05, 1.95], [expected_values[1]] * 2, "b", alpha=0.75, label="expectation male")
    ax.bar([0.5, 1.5], values, width=0.9, color=["r", "b"], alpha=0.25, yerr=errors, ecolor="grey", capsize=6, ec="k")
    ax.set_xticks([0.5, 1.5], labels=["female", "male"])
    max_y = np.max(values)
    if errors:
        max_y += np.max(errors)
    ax.set_ylim(0, 1.25 * max_y)
    ax.set_ylabel(axis_label)
    ax.legend()
