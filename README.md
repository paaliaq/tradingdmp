# tradingdmp

This repo defines classes for data, models and policies (dmp). These classes are all
meant to be used by our trading applications.

## Project Organization

```
├── notebooks               <- Notebooks named like `YYYYMMDD-filename.ipynb`
├── src                     <- Package of this project
│   ├── data                <- Subpackage for data pipeline
│   ├── models              <- Subpackage for models
│   └── policy              <- Subpackage for policies
├── data                    <- Repository with data for development and testing.
│   ├── raw                 <- The original, immutable data
│   ├── interim             <- Intermediate data that has been transformed
│   └── final               <- The final data sets for modeling
├── models                  <- Trained and serialized models, model predictions, or model summaries
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures             <- Generated graphics and figures to be used in reporting
│   └── tables              <- Generated tables to be used in reporting
├── references              <- Data dictionaries, manuals, and all other explanatory materials
├── tests                   <- Unit tests for src/ and example tests for scripts/
├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
├── .config                 <- All config.json files belong in this folder
├── README.md               <- The top-level README for developers using this project
├── pyproject.toml          <- Specifies all requirements for this project using poetry
├── requirements.txt        <- Generated with `poetry export -f requirements.txt --output requirements.txt`
├── .gitlab-ci.yml          <- CI pipeline for code and documentation format checks
└── setup.cfg               <- Config file for pipeline
```

## How to run the pipeline linters locally

The pipeline (`.gitlab-ci.yml`) also contains some checks to ensure code quality. You can run these checks locally with the following commands from the main directory. You can also run the commands separately if that's what you prefer.

```shell
black --check src/ && flake8 src/ && mypy src/ && pydocstyle src/ && darglint src/
```