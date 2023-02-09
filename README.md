# learn2thermML
Machine learning of low and high temperature proteins

## Getting started
### Environment
Create and activate the environment specified in `environment.yml`

```
conda env create --file environment.yml
conda activate learn2thermML
```

Ensure that the following environmental variables are set for pipeline exacution:  
- `LOGLEVEL` (optional) - Specified logging level to run the package. eg 'INFO' or 'DEBUG'

### Execution
Data Version Control (DVC) is used to track data, parameters, metrics, and execution pipelines.

To use a DVC remote, see the the [documentation](https://dvc.org/doc/command-reference/remote).

DVC tracked data, metrics, and models are found in `./data` while scripts and parameters can be found in `./pipeline`. To execute pipeline steps, run `dvc repro <stage-name>` where stages are listed below:

- TODO

Note that script execution is expected to occur with the top level as the current working directory, and paths are specified with respect to the repo top level.

### Python package
Installable, importable code is found in `learn2therm` and should be installed given the above steps in the __Environemnt__ section.

## Directory
```
-data/                                      # Contains DVC tracked data, models, and metrics
-pipeline/                                  # Contains DVC tracked executable pipeline steps and parameters
-notebooks/                                 # notebooks for testing and decision making
-environment.yml                            # Conda dependancies
-docs/                                      # repository documentation
```