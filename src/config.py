"""
Defines all important configs and paths
"""
from pathlib import Path
import matplotlib.pyplot as plt


root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"
notebooks_dir = root_dir / "notebooks"
figures_dir = notebooks_dir / "figures"
figures_dir.mkdir(exist_ok=True)


def get_save_figure_function(notebook_name):
    """
    exercise_number, task_number -> ints that define for which exercise and task to save the figures for

    return -> function that will save figure in right directory with right prefix and extension, requiring only one name
    """
    def save_current_figure(name):
        file_name = notebook_name + "_" + name + ".pdf"
        plt.savefig(figures_dir / file_name, bbox_inches="tight")

    return save_current_figure
