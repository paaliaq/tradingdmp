# tradingapp

This repo defines core classes of our trading applications.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── base_data.py
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── base_model.py
    │   └── policy         <- Scripts to get actions to be taken by the trading application
    │       └── base_policy.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Template reference: 

- https://github.com/fedejaure/cookiecutter-modern-pypackage
- https://github.com/drivendata/cookiecutter-data-science/

## Deployment reference: 

The main idea is to create a tox.ini with all linter commands (flake8, mypy, darglint). Then, our gitlab-ci.yml is configured to run the tox file.

- tox: 
    - https://tox.readthedocs.io/en/latest/
    - https://tox.readthedocs.io/en/latest/example/basic.html
    - https://www.integralist.co.uk/posts/toxini/#example-tox-ini
- documentation:
    - darglint: https://github.com/terrencepreilly/darglint
    - google docstrings: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
    - sphinx: https://github.com/sphinx-doc/sphinx 
- typing:
    - mypy:https://github.com/python/mypy
    - typing: https://docs.python.org/3/library/typing.html
- pep-8:
    - flake8: https://github.com/PyCQA/flake8
    - (pylint): https://github.com/PyCQA/pylint
- examples:
    - tox in gitlab-ci.yml: https://stackoverflow.com/questions/62568690/run-tox-and-docker-compose-in-gitlab-ci-yml-failed-connection-to-localhost