## parametric-resonance

## structure

```
.
|--main.py         # entry-point script to run experiments
|--experiments/     
|  |--run_01/     # an example experiment, with sample config files
|  |  |--in/       # location for config files for an experiment
|  |  |--out/      # location for plots, saved data, etc.
|--src/            # codebase
```

## setup
first, set up python environment. then, initialise and install all necessary packages and dependencies with the following commands:

```
> cd ~
> python -m venv $VENV_DIR             # replace $VENV_DIR with the folder where you want your virtual environment set up
> source $VENV_DIR/bin/activate
> cd $THIS_DIR                         # replace $THIS_DIR with wherever you have downloaded this code
> pip install -r requirements.txt      # this will automatically install all dependent packages
```

## usage
the entry-point to interacting with the codebase is the script `main.py`, which is invoked as follows
```
> python main.py setup -D $EXPERIMENT_NAME  # sets up the experiment folder under experiment/$EXPERIMENT_NAME
                                            # note: after this step, user must MANUALLY move config files to experiment/$EXPERIMENT_NAME/in
> python main.py run   -D $EXPERIMENT_NAME  # will solve the axion field ODE and integrate to find F(u,v) for a range of u-values
> python main.py plot  -D $EXPERIMENT_NAME  # generates some nice plots of axion field strength and correlator.
```
