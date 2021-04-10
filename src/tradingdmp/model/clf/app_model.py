"""Application classes for models that are used in our trading apps."""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from supervised.automl import AutoML
from tradingdmp.model.clf.base_model import BaseFeatureModel


class MljarAutoMl(BaseFeatureModel):
    """Base class for feature based modeling.

    The main difference between BaseFeatureModel and BaseTimeModel is that
    BaseFeatureModel uses data from BaseFeatureData, whereas BaseTimeModel uses data
    from BaseTimeData.
    """

    def __init__(
        self,
        mode: str = "Compete",
        ml_task: str = "auto",
        algorithms: Union[str, List[str]] = "auto",
        train_ensemble: bool = True,
        stack_models: Union[str, bool] = "auto",
        eval_metric: str = "auto",
        validation_strategy: Union[str, dict] = "auto",
        explain_level: Union[str, int] = "auto",
        golden_features: Union[bool, str] = "auto",
        features_selection: Union[bool, str] = "auto",
        start_random_models: Union[int, str] = "auto",
        hill_climbing_steps: Union[int, str] = "auto",
        top_models_to_improve: Union[int, str] = "auto",
        boost_on_errors: Union[bool, str] = "auto",
        kmeans_features: Union[bool, str] = "auto",
        mix_encoding: Union[bool, str] = "auto",
        optuna_init_params: dict = {},
        optuna_verbose: bool = True,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 12345,
        results_path: Optional[str] = None,
        model_time_limit: Optional[int] = None,
        total_time_limit: Optional[int] = 10800,
        max_single_prediction_time: Optional[Union[int, float]] = None,
        optuna_time_budget: Optional[int] = None,
    ) -> None:
        """Initializes model for MljarAutoMl class.

        Please see mljar documentation for argument reference or see the Args reference
        below (which is a copy of the mljar Args reference).
        - https://github.com/mljar/mljar-supervised/blob/master/supervised/automl.py
        - https://supervised.mljar.com/api/#supervised.automl.AutoML.predict_proba

        Args:
            mode (str): Can be {`Explain`, `Perform`, `Compete`, `Optuna`}. This
            parameter defines the goal of AutoML and how intensive the AutoML search
            will be.
                - `Explain` : To to be used when the user wants to explain and
                    understand the data.
                    - Uses 75%/25% train/test split.
                    - Uses the following models: `Baseline`, `Linear`, `Decision Tree`,
                        `Random Forest`, `XGBoost`, `Neural Network`, and `Ensemble`.
                    - Has full explanations in reports: learning curves, importance
                        plots, and SHAP plots.
                - `Perform` : To be used when the user wants to train a model that will
                    be used in real-life use cases.
                    - Uses 5-fold CV (Cross-Validation).
                    - Uses the following models: `Linear`, `Random Forest`, `LightGBM`,
                        `XGBoost`, `CatBoost`, `Neural Network`, and `Ensemble`.
                    - Has learning curves and importance plots in reports.
                - `Compete` : To be used for machine learning competitions (maximum
                    performance).
                    - Uses 80/20 train/test split, or 5-fold CV, or 10-fold CV
                        (Cross-Validation) - it depends on `total_time_limit`. If not
                        set directly, AutoML will select validation automatically.
                    - Uses the following models: `Decision Tree`, `Random Forest`,
                    `Extra Trees`, `LightGBM`,  `XGBoost`, `CatBoost`, `Neural Network`,
                        `Nearest Neighbors`, `Ensemble`, and `Stacking`.
                    - It has only learning curves in the reports.
                - `Optuna` : To be used for creating highly-tuned machine learning
                    models.
                    - Uses 10-fold CV (Cross-Validation).
                    - It tunes with Optuna the following algorithms: `Random Forest`,
                        `Extra Trees`, `LightGBM`, `XGBoost`, `CatBoost`,
                        `Neural Network`.
                    - It applies `Ensemble` and `Stacking` for trained models.
                    - It has only learning curves in the reports.
            ml_task (str): Can be {"auto", "binary_classification",
                "multiclass_classification", "regression"}.
                - If left `auto` AutoML will try to guess the task based on target
                    values.
                - If there will be only 2 values in the target, then task will be set to
                    `"binary_classification"`.
                - If number of values in the target will be between 2 and 20 (included),
                    then task will be set to `"multiclass_classification"`.
                - In all other casses, the task is set to `"regression"`.
            algorithms (list of str): The list of algorithms that will be used in the
                training. The algorithms can be:
                - `Baseline`,
                - `Linear`,
                - `Decision Tree`,
                - `Random Forest`,
                - `Extra Trees`,
                - `LightGBM`,
                - `Xgboost`,
                - `CatBoost`,
                - `Neural Network`,
                - `Nearest Neighbors`,
            train_ensemble (boolean): Whether an ensemble gets created at the end of the
                training.
            stack_models (boolean): Whether a models stack gets created at the end of
                the training. Stack level is 1.
            eval_metric (str): The metric to be used in early stopping and to compare
                models.
                - for binary classification: `logloss`, `auc`, `f1`,
                    `average_precision`, `accuracy` - default is logloss (if left
                    "auto")
                - for mutliclass classification: `logloss`, `f1`, `accuracy` - default
                    is `logloss` (if left "auto")
                - for regression: `rmse`, `mse`, `mae`, `r2`, `mape`, `spearman`,
                    `pearson` - default is `rmse` (if left "auto")
            validation_strategy (dict): Dictionary with validation type. Right now
                train/test split and cross-validation are supported.
                - Example:
                    Cross-validation exmaple:
                    {
                        "validation_type": "kfold",
                        "k_folds": 5,
                        "shuffle": True,
                        "stratify": True,
                        "random_seed": 123
                    }
                    Train/test example:
                    {
                        "validation_type": "split",
                        "train_ratio": 0.75,
                        "shuffle": True,
                        "stratify": True
                    }
            explain_level (int): The level of explanations included to each model:
                - if `explain_level` is `0` no explanations are produced.
                - if `explain_level` is `1` the following explanations are produced:
                    importance plot (with permutation method), for decision trees
                    produce tree plots, for linear models save coefficients.
                - if `explain_level` is `2` the following explanations are produced: the
                    same as `1` plus SHAP explanations.
                If left `auto` AutoML will produce explanations based on the selected
                `mode`.
            golden_features (boolean): Whether to use golden features
                If left `auto` AutoML will use golden features based on the selected
                `mode`:
                - If `mode` is "Explain", `golden_features` = False.
                - If `mode` is "Perform", `golden_features` = True.
                - If `mode` is "Compete", `golden_features` = True.
            features_selection (boolean): Whether to do features_selection
                If left `auto` AutoML will do feature selection based on the selected
                `mode`:
                - If `mode` is "Explain", `features_selection` = False.
                - If `mode` is "Perform", `features_selection` = True.
                - If `mode` is "Compete", `features_selection` = True.
            start_random_models (int): Number of starting random models to try.
                If left `auto` AutoML will select it based on the selected `mode`:
                - If `mode` is "Explain", `start_random_models` = 1.
                - If `mode` is "Perform", `start_random_models` = 5.
                - If `mode` is "Compete", `start_random_models` = 10.
            hill_climbing_steps (int): Number of steps to perform during hill climbing.
                If left `auto` AutoML will select it based on the selected `mode`:
                - If `mode` is "Explain", `hill_climbing_steps` = 0.
                - If `mode` is "Perform", `hill_climbing_steps` = 2.
                - If `mode` is "Compete", `hill_climbing_steps` = 2.
            top_models_to_improve (int): Number of best models to improve in
                `hill_climbing` steps. If left `auto` AutoML will select it based on the
                selected `mode`:
                - If `mode` is "Explain", `top_models_to_improve` = 0.
                - If `mode` is "Perform", `top_models_to_improve` = 2.
                - If `mode` is "Compete", `top_models_to_improve` = 3.
            boost_on_errors (boolean): Whether a model with boost on errors from
                previous best model should be trained. By default available in the
                `Compete` mode.
            kmeans_features (boolean): Whether a model with k-means generated features
            should be trained. By default available in the `Compete` mode.
            mix_encoding (boolean): Whether a model with mixed encoding should be
            trained. Mixed encoding is the encoding that uses label encoding
                for categoricals with more than 25 categories, and one-hot binary
                encoding for other categoricals. It is only applied if there are
                categorical features with cardinality smaller than 25. By default it is
                available in the `Compete` mode.
            optuna_init_params (dict): If you have already tuned parameters from Optuna
                you can reuse them by setting this parameter. This parameter is only
                used when `mode="Optuna"`. The dict should have structure and params as
                specified in the MLJAR AutoML .
            optuna_verbose (boolean): If true the Optuna tuning details are displayed.
                Set to `True` by default.
            n_jobs (int): Number of CPU cores to be used. By default is set to `-1`
                which means using  all processors.
            verbose (int): Controls the verbosity when fitting and predicting. Note:
                Still not implemented, please left `1`
            random_state (int): Controls the randomness of the `AutoML`
            results_path (str): The path with results. If None, then the name of
            directory will be generated with the template: AutoML_{number},
                where the number can be from 1 to 1,000 - depends which direcory name
                will be available. If the `results_path` will point to directory with
                AutoML results (`params.json` must be present), then all models will be
                loaded.
            model_time_limit (int): The time limit for training a single model, in
                seconds. If `model_time_limit` is set, the `total_time_limit` is not
                respected. The single model can contain several learners. The time limit
                for subsequent learners is computed based on `model_time_limit`. For
                example, in the case of 10-fold cross-validation, one model will have
                10 learners. The `model_time_limit` is the time for all 10 learners.
            total_time_limit (int): The total time limit in seconds for AutoML training.
                It is not used when `model_time_limit` is not `None`.
            max_single_prediction_time (int or float): The limit for prediction time for
                single sample. Use it if you want to have a model with fast predictions.
                Ideal for creating ML pipelines used as REST API. Time is in seconds. By
                default (`max_single_prediction_time=None`) models are not optimized for
                fast predictions, except the mode `Perform`. For the mode `Perform` the
                default is `0.5` seconds.
            optuna_time_budget (int): The time in seconds which should be used by Optuna
                to tune each algorithm. It is time for tuning single algorithm.
                If you select two algorithms: Xgboost and CatBoost, and set
                optuna_time_budget=1000, then Xgboost will be tuned for 1000 seconds and
                CatBoost will be tuned for 1000 seconds. What is more, the tuning is
                made for each data type, for example for raw data and for data with
                inserted Golden Features. This parameter is only used when
                `mode="Optuna"`. If you set `mode="Optuna"` and forget to set this
                parameter, it will be set to 3600 seconds.
        """
        self.model = AutoML(
            results_path=results_path,
            total_time_limit=total_time_limit,
            mode=mode,
            ml_task=ml_task,
            model_time_limit=model_time_limit,
            algorithms=algorithms,
            train_ensemble=train_ensemble,
            stack_models=stack_models,
            eval_metric=eval_metric,
            validation_strategy=validation_strategy,
            explain_level=explain_level,
            golden_features=golden_features,
            features_selection=features_selection,
            start_random_models=start_random_models,
            hill_climbing_steps=hill_climbing_steps,
            top_models_to_improve=top_models_to_improve,
            boost_on_errors=boost_on_errors,
            kmeans_features=kmeans_features,
            mix_encoding=mix_encoding,
            max_single_prediction_time=max_single_prediction_time,
            optuna_time_budget=optuna_time_budget,
            optuna_init_params=optuna_init_params,
            optuna_verbose=optuna_verbose,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
        """Method for fitting a model.

        This function should fit the model given training data x and y. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            x: Training features in data frame of shape (n, m), where n is the number of
                samples and m is the number of features.
            y: Training targets in numpy array of shape (n, 1), where n is the number of
                samples. y needs to contain classes in string format.
            *args: Variable length argument list. Not used.
            **kwargs: Arbitrary keyword arguments. Not used.

        Returns:
            None
        """
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame, *args: Any, **kwargs: Any) -> np.ndarray:
        """Method for predicting classes with a fitted model.

        This function should predict with a model given test data x. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            x: Test features in data frame of shape (n, m), where n is the number of
                samples and m is the number of features.
            *args: Variable length argument list. Not used.
            **kwargs: Arbitrary keyword arguments. Not used.

        Returns:
            y: Predicted targets in a numpy array of shape (n, 1), where n is the number
                of samples. y will contain predicted classes in string format.
        """
        return self.model.predict(x)

    def predict_proba(self, x: pd.DataFrame, *args: Any, **kwargs: Any) -> np.ndarray:
        """Method for predicting class probabilities with a fitted model.

        This function should predict with a model given test data x. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            x: Test features in data frame of shape (n, m), where n is the number of
                samples and m is the number of features.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            y: Predicted targets in a numpy array of shape (n, d), where n is the number
                of samples and d is the number of classes. y will contain predicted
                class probabilities for each observation and class in float format.
        """
        return self.model.predict_proba(x)
