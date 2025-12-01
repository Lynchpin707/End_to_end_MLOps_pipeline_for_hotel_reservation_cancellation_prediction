import logging
import pandas as pd
from zenml import step

from src.model_dev import LogisticRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
    ) -> RegressorMixin:
    """Trains the model on the ingested data

    Args:
        X_train (pd.DataFrame)
        X_test (pd.DataFrame)
        y_train (pd.DataFrame)
        y_test (pd.DataFrame)

    Returns:
        RegressorMixin: the trained model
    """

    model = None
    if config.model_name == "LogisticRegressionModel":
        model = LogisticRegressionModel()
        train_model = model.train(X_train, y_train)
        return train_model
    else:
        raise ValueError(f"Model {config.model_name} not supported")
        
    