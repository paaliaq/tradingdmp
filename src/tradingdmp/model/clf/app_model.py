"""Application classes for models that are used in our trading apps."""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext
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

    def predict(
        self, context: PythonModelContext, x: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Method for predicting classes with a fitted model.

        This function should predict with a model given test data x. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            context: A PythonModelContext instance containing artifacts that the model
                can use to perform inference.
            x: Test features in data frame of shape (n, m), where n is the number of
                samples and m is the number of features.
            *args: Variable length argument list. Not used.
            **kwargs: Arbitrary keyword arguments. Not used.

        Returns:
            y: Predicted targets in a numpy array of shape (n, 1), where n is the number
                of samples. y will contain predicted classes in string format.
        """
        return self.model.predict(x)

    def predict_proba(
        self, context: PythonModelContext, x: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Method for predicting class probabilities with a fitted model.

        This function should predict with a model given test data x. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            context: A PythonModelContext instance containing artifacts that the model
                can use to perform inference.
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
