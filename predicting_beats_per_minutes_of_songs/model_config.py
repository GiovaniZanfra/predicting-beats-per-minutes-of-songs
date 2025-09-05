"""
Model configuration for BPM prediction.

This module defines available models and their hyperparameters.
You can easily switch between models by changing the MODEL_NAME in the config.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

# Optional imports for advanced models
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_class: Any
    hyperparameters: Dict[str, Any]
    description: str = ""


# Available models configuration
AVAILABLE_MODELS = {
    "xgboost": ModelConfig(
        name="XGBoost",
        model_class=xgb.XGBRegressor,
        hyperparameters={
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
            "enable_categorical": True,
            "eval_metric": "rmse"
        },
        description="XGBoost regressor with optimized hyperparameters"
    ),
    
    "random_forest": ModelConfig(
        name="Random Forest",
        model_class=RandomForestRegressor,
        hyperparameters={
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        description="Random Forest regressor with optimized hyperparameters"
    ),
    
    "ridge": ModelConfig(
        name="Ridge Regression",
        model_class=Ridge,
        hyperparameters={
            "alpha": 1.0,
            "random_state": 42
        },
        description="Ridge regression with L2 regularization"
    ),
    
    "lasso": ModelConfig(
        name="Lasso Regression",
        model_class=Lasso,
        hyperparameters={
            "alpha": 0.1,
            "random_state": 42,
            "max_iter": 2000
        },
        description="Lasso regression with L1 regularization"
    ),
    
    "elastic_net": ModelConfig(
        name="Elastic Net",
        model_class=ElasticNet,
        hyperparameters={
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "random_state": 42,
            "max_iter": 2000
        },
        description="Elastic Net with L1 and L2 regularization"
    ),
    
    "svr": ModelConfig(
        name="Support Vector Regression",
        model_class=SVR,
        hyperparameters={
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "epsilon": 0.1
        },
        description="Support Vector Regression with RBF kernel"
    )
}

# Add optional models if available
if LGBMRegressor is not None:
    AVAILABLE_MODELS["lightgbm"] = ModelConfig(
        name="LightGBM",
        model_class=LGBMRegressor,
        hyperparameters={
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
            "verbose": -1
        },
        description="LightGBM regressor with optimized hyperparameters"
    )

if CatBoostRegressor is not None:
    AVAILABLE_MODELS["catboost"] = ModelConfig(
        name="CatBoost",
        model_class=CatBoostRegressor,
        hyperparameters={
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 50
        },
        description="CatBoost regressor with optimized hyperparameters"
    )


# Current model selection - Change this to switch models
CURRENT_MODEL = "xgboost"  # Options: xgboost, lightgbm, catboost, random_forest, ridge, lasso, elastic_net, svr

# Cross-validation configuration
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42
}

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "scoring": "neg_mean_squared_error"
}


def get_model_config(model_name: Optional[str] = None) -> ModelConfig:
    """Get model configuration by name."""
    if model_name is None:
        model_name = CURRENT_MODEL
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    return AVAILABLE_MODELS[model_name]


if __name__ == "__main__":
    list_available_models()
