from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from predicting_beats_per_minutes_of_songs.config import MODELS_DIR, PROCESSED_DATA_DIR
from predicting_beats_per_minutes_of_songs.model_config import get_model_config, CV_CONFIG

app = typer.Typer()


def evaluate_model(y_true, y_pred, dataset_name="Validation"):
    """Evaluate model performance and log metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"{dataset_name} Metrics:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  R²:   {r2:.4f}")
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def perform_kfold_validation(X, y, model_config, cv_config):
    """Perform KFold cross-validation and return out-of-fold predictions."""
    logger.info(f"Starting {cv_config['n_splits']}-fold cross-validation...")
    
    kf = KFold(
        n_splits=cv_config['n_splits'],
        shuffle=cv_config['shuffle'],
        random_state=cv_config['random_state']
    )
    
    oof_predictions = np.zeros(len(X))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=cv_config['n_splits'], desc="KFold")):
        logger.info(f"Training fold {fold + 1}/{cv_config['n_splits']}")
        
        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize model
        model = model_config.model_class(**model_config.hyperparameters)
        
        # Train model
        if hasattr(model, 'fit'):
            model.fit(X_train_fold, y_train_fold)
        else:
            raise ValueError(f"Model {model_config.name} does not have fit method")
        
        # Make predictions
        val_pred = model.predict(X_val_fold)
        oof_predictions[val_idx] = val_pred
        
        # Evaluate fold
        fold_metrics = evaluate_model(y_val_fold, val_pred, f"Fold {fold + 1}")
        fold_scores.append(fold_metrics)
        
        logger.info(f"Fold {fold + 1} RMSE: {fold_metrics['rmse']:.4f}")
    
    # Calculate overall CV metrics
    cv_metrics = evaluate_model(y, oof_predictions, "Cross-Validation")
    
    # Calculate mean and std of fold scores
    mean_rmse = np.mean([score['rmse'] for score in fold_scores])
    std_rmse = np.std([score['rmse'] for score in fold_scores])
    
    logger.info(f"CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    return oof_predictions, cv_metrics, fold_scores


def train_final_model(X, y, model_config):
    """Train final model on entire training set."""
    logger.info("Training final model on entire training set...")
    
    # Initialize and train model
    final_model = model_config.model_class(**model_config.hyperparameters)
    final_model.fit(X, y)
    
    logger.success("Final model training complete!")
    return final_model


def predict_test_set(model, X_test, test_ids):
    """Make predictions on test set."""
    logger.info("Making predictions on test set...")
    
    test_predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': test_predictions
    })
    
    logger.info(f"Generated {len(test_predictions)} test predictions")
    logger.info(f"Prediction range: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
    
    return submission


@app.command()
def main(
    train_features_path: Path = PROCESSED_DATA_DIR / "train_features.parquet",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.parquet",
    model_name: str = typer.Option(None, help="Model name to use (overrides config)"),
    model_path: Path = MODELS_DIR / "model.pkl",
    oof_predictions_path: Path = PROCESSED_DATA_DIR / "oof_predictions.csv",
    test_predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    save_model: bool = typer.Option(True, help="Whether to save the trained model"),
):
    """Train model with KFold validation and generate predictions."""
    
    # Get model configuration
    model_config = get_model_config(model_name)
    logger.info(f"Using model: {model_config.name}")
    logger.info(f"Model description: {model_config.description}")
    
    # Load training data
    logger.info(f"Loading training features from {train_features_path}")
    if train_features_path.suffix == '.parquet':
        train_df = pd.read_parquet(train_features_path)
    else:
        train_df = pd.read_csv(train_features_path)
    
    # Separate features and target
    target_col = 'BeatsPerMinute'
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    X_train = train_df.drop(columns=[target_col, 'id'] if 'id' in train_df.columns else [target_col])
    y_train = train_df[target_col]
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Features: {list(X_train.columns)}")
    
    # Perform KFold validation
    oof_predictions, cv_metrics, fold_scores = perform_kfold_validation(
        X_train, y_train, model_config, CV_CONFIG
    )
    
    # Save out-of-fold predictions
    oof_df = pd.DataFrame({
        'id': train_df['id'] if 'id' in train_df.columns else range(len(train_df)),
        'BeatsPerMinute': oof_predictions
    })
    oof_df.to_csv(oof_predictions_path, index=False)
    logger.info(f"Saved out-of-fold predictions to {oof_predictions_path}")
    
    # Train final model on entire training set
    final_model = train_final_model(X_train, y_train, model_config)
    
    # Save model if requested
    if save_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"Saved model to {model_path}")
    
    # Load test data and make predictions
    logger.info(f"Loading test features from {test_features_path}")
    if test_features_path.suffix == '.parquet':
        test_df = pd.read_parquet(test_features_path)
    else:
        test_df = pd.read_csv(test_features_path)
    
    # Ensure test data has same features as training data
    test_features = test_df.drop(columns=['id'] if 'id' in test_df.columns else [])
    
    # Align columns with training data
    test_features = test_features.reindex(columns=X_train.columns, fill_value=0)
    
    logger.info(f"Test data shape: {test_features.shape}")
    
    # Make test predictions
    test_submission = predict_test_set(
        final_model, 
        test_features, 
        test_df['id'] if 'id' in test_df.columns else range(len(test_df))
    )
    
    # Save test predictions
    test_submission.to_csv(test_predictions_path, index=False)
    logger.info(f"Saved test predictions to {test_predictions_path}")
    
    # Final summary
    logger.success("Training pipeline complete!")
    logger.info(f"Final CV RMSE: {cv_metrics['rmse']:.4f}")
    logger.info(f"Final CV R²: {cv_metrics['r2']:.4f}")


if __name__ == "__main__":
    app()
