# Pedestrian Crowd Modelling Using a Cellular Automaton

We recommend using an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/) or [VSCode](https://code.visualstudio.com/) to edit the python files.

To manage the python package installations, we recommend using a virtual environment like [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/miniconda.html).

## Environment Installation
### conda

(Assumes conda is already installed). In a terminal, in the project root directory, run:

```conda env create --file=environment.yml```

and the conda environment will be created with the given requirements. 
To activate it, run: 

```conda activate cmca``` 

and configure your IDE to use this conda environment as well.

### venv

Follow the instructions [here](https://docs.python.org/3/library/venv.html) to create and activate a new virtual python environment. 
Then install the required packages: In a terminal, with the environment activated, run:

```python -m pip install requirements.txt -r```

and configure your IDE to use this environment.

## Notebooks

To launch the jupyter notebooks in a browser, simply run:

```jupyter notebook```

in a terminal, in the project root directory, with the python environment activated. Then you can navigate to the desired notebook and edit it using the jupyter browser interface.
