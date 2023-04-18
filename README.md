Machine Learning Application for Type Identification based on Image Data
==============================

Author: Daniel Lam
A ML-project with data versioning (dvc).

## Requirements

Please make sure you have these available on the environment where these scripts
will run:

- [Git](https://git-scm.com/)
- [Python](https://www.python.org/) 3.8+ (with `python` and [pip](https://pypi.org/project/pip/) commands)
- [DVC](https://dvc.org)

This instruction assumes that DVC is already installed, as it is frequently
used as a global tool like Git. If DVC is not installed, see 
the [DVC installation guide](https://dvc.org/doc/install) on how to install DVC.
The DVC version used is 2.17.0.

## Installation

Clone the repository.

```console
$ git clone https://github.com/Daesser/ML-classifier-dvc.git
$ cd belt-classification
```
It is recommended to create a virtual environment with a tool such as [virtualenv](https://virtualenv.pypa.io/en/stable/):

```console
$ virtualenv -p python .venv
$ source .venv/bin/activate
```
or [anaconda](https://docs.anaconda.com/):

```console
$ conda create --name myenv
$ conda activate myenv
```

Install the requirements and import the src

```console
$ pip install -r requirements.txt
$ pip install .
```

## Getting the data

DVC project can be preconfigured with DVC
[remote storage](https://dvc.org/doc/commands-reference/remote) that holds raw
data (input) by adding the remote: `dvc remote add -d myremote /path/to/remote`.
In this case running [`dvc pull`](https://man.dvc.org/pull) will download the data

```console
$ dvc pull
```

The preprocessing step can be skipped (or must be adjusted) for other data organizations.

## Running the code

Run [`dvc repro`](https://man.dvc.org/repro) to start the
[pipeline](https://dvc.org/doc/commands-reference/pipeline):

```console
$ dvc repro
```
One can also run the stages of the pipeline individually (without DVC), e.g, `python src/data/preprocess.py`.
The stages are:

- `python src/data/validate.py`: Validates the Dataset.
- `python src/data/preprocess.py`: Preprocesses the data.
- `python src/data/analyze.py`: Analyzes the Dataset.
- `python src/models/train_model.py`: Trains a ML-model.
- `python src/evaluation/evaluate.py`: Evaluation and plots (validation set).
- `python src/evaluation/evaluate_test.py`: Evaluation and plots (test set).
- `python src/test/behavior_test.py`: Runs behavior tests.

## DVC Studio
This repository can be linked to [Iterative Studio](https://studio.iterative.ai/) easily.
Iterative Studio is a online web application (or host on-premise) which enables data 
and machine learning models management, run and track experiments, and visualize and share
results. 

## Debugging

You might run into this error: AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'.
In this case you have to upgrade [pip](https://pypi.org/project/pip/) and [pyopenssl](https://pypi.org/project/pyOpenSSL/)

```console
$ pip install pip --upgrade
$ pip install pyopenssl --upgrade
```

DVC is changing very fast, new features are added frequently and some parts of the
code has to be rewritten with the newer versions. 
The actual version is 2.45.1 (4.3.2023). Please refer to https://dvc.org/doc.

## Project Structure

------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── report             <- Generated analysis as png, csv etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting.
    │   ├── metrics        <- Metrics during training process.
    │   └── plots          <- Confusion matrices.
    │
    ├── reports            <- Some reports of previous experiments.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data           <- Scripts to generate datasets.
    │   ├── models         <- Scripts to build and train models.
    │   └── evaluation     <- Scripts to evaluate models.
    │
    ├── dvc.yaml           <- defines stages of pipeline.
    │
    ├── params.yaml        <- specifies (hyper)parameters or config file.
    │
    └── test_environment.py

--------

