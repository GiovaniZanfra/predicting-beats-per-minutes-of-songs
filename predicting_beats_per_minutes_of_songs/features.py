from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from predicting_beats_per_minutes_of_songs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ratio/proportion features."""
    logger.info("Creating ratio/proportion features...")
    
    # AudioLoudness adjusted by track duration (avoiding log(0))
    df['AudioLoudness_per_duration'] = df['AudioLoudness'] / np.log(df['TrackDurationMs'] + 1)
    
    # Energy / AcousticQuality (avoiding division by zero)
    df['Energy_per_AcousticQuality'] = df['Energy'] / (df['AcousticQuality'] + 1e-8)
    
    # RhythmScore / MoodScore (avoiding division by zero)
    df['RhythmScore_per_MoodScore'] = df['RhythmScore'] / (df['MoodScore'] + 1e-8)
    
    return df


def create_multiplicative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create multiplicative interaction features."""
    logger.info("Creating multiplicative interaction features...")
    
    # Energy interactions
    df['Energy_x_RhythmScore'] = df['Energy'] * df['RhythmScore']
    df['Energy_x_AudioLoudness'] = df['Energy'] * df['AudioLoudness']
    
    # Vocal vs Instrumental interactions
    df['VocalContent_x_InstrumentalScore'] = df['VocalContent'] * df['InstrumentalScore']
    
    # Mood and Live performance interactions
    df['MoodScore_x_LivePerformanceLikelihood'] = df['MoodScore'] * df['LivePerformanceLikelihood']
    
    # Additional meaningful interactions
    df['RhythmScore_x_Energy'] = df['RhythmScore'] * df['Energy']
    df['AcousticQuality_x_MoodScore'] = df['AcousticQuality'] * df['MoodScore']
    
    return df


def create_nonlinear_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Create non-linear transformations."""
    logger.info("Creating non-linear transformations...")
    
    # Square transformations
    df['Energy_squared'] = df['Energy'] ** 2
    df['RhythmScore_squared'] = df['RhythmScore'] ** 2
    df['MoodScore_squared'] = df['MoodScore'] ** 2
    
    # Square root transformations (ensuring non-negative values)
    df['AudioLoudness_sqrt'] = np.sqrt(df['AudioLoudness'] - df['AudioLoudness'].min() + 1)
    df['Energy_sqrt'] = np.sqrt(df['Energy'])
    df['RhythmScore_sqrt'] = np.sqrt(df['RhythmScore'])
    
    # Logarithmic transformations
    df['TrackDurationMs_log'] = np.log(df['TrackDurationMs'] + 1)
    df['AudioLoudness_log'] = np.log(df['AudioLoudness'] - df['AudioLoudness'].min() + 1)
    
    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical/discretization features."""
    logger.info("Creating cyclical/discretization features...")
    
    # Duration bins (short <180s, medium 180-300s, long >300s)
    df['TrackDuration_bins'] = pd.cut(
        df['TrackDurationMs'] / 1000,  # Convert to seconds
        bins=[0, 180, 300, float('inf')],
        labels=['short', 'medium', 'long']
    )
    
    # Duration bin indicators
    df['TrackDuration_short'] = (df['TrackDurationMs'] / 1000 < 180).astype(int)
    df['TrackDuration_medium'] = ((df['TrackDurationMs'] / 1000 >= 180) & 
                                  (df['TrackDurationMs'] / 1000 <= 300)).astype(int)
    df['TrackDuration_long'] = (df['TrackDurationMs'] / 1000 > 300).astype(int)
    
    # Note: BeatsPerMinute is the target variable, so we don't create features from it
    
    return df


def create_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create difference/contrast features."""
    logger.info("Creating difference/contrast features...")
    
    # Vocal vs Instrumental dominance
    df['VocalContent_minus_InstrumentalScore'] = df['VocalContent'] - df['InstrumentalScore']
    
    # Acoustic vs Energy contrast (smooth vs aggressive)
    df['AcousticQuality_minus_Energy'] = df['AcousticQuality'] - df['Energy']
    
    # Mood vs Rhythm contrast
    df['MoodScore_minus_RhythmScore'] = df['MoodScore'] - df['RhythmScore']
    
    # Live vs Studio contrast
    df['LivePerformanceLikelihood_minus_AcousticQuality'] = (
        df['LivePerformanceLikelihood'] - df['AcousticQuality']
    )
    
    return df


def create_duration_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """Create duration-based normalizations."""
    logger.info("Creating duration-based normalizations...")
    
    # Duration-normalized features (always available)
    df['RhythmScore_per_duration'] = df['RhythmScore'] * df['TrackDurationMs'] / 1000
    df['Energy_per_duration'] = df['Energy'] * df['TrackDurationMs'] / 1000
    df['MoodScore_per_duration'] = df['MoodScore'] * df['TrackDurationMs'] / 1000
    
    # Note: BeatsPerMinute is the target variable, so we don't create features from it
    
    return df


def create_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create polynomial features for key variables."""
    logger.info("Creating polynomial features...")
    
    # Polynomial features for highly correlated variables with BPM
    # Energy and RhythmScore are likely highly correlated with BPM
    df['Energy_x_RhythmScore_squared'] = df['Energy'] * (df['RhythmScore'] ** 2)
    df['RhythmScore_x_Energy_squared'] = df['RhythmScore'] * (df['Energy'] ** 2)
    
    # Cross-polynomials
    df['Energy_squared_x_RhythmScore_squared'] = (df['Energy'] ** 2) * (df['RhythmScore'] ** 2)
    
    return df


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create statistical summary features."""
    logger.info("Creating statistical features...")
    
    # Audio feature combinations
    audio_features = ['Energy', 'RhythmScore', 'MoodScore', 'AcousticQuality']
    df['Audio_features_mean'] = df[audio_features].mean(axis=1)
    df['Audio_features_std'] = df[audio_features].std(axis=1)
    df['Audio_features_max'] = df[audio_features].max(axis=1)
    df['Audio_features_min'] = df[audio_features].min(axis=1)
    
    # Content feature combinations
    content_features = ['VocalContent', 'InstrumentalScore', 'LivePerformanceLikelihood']
    df['Content_features_mean'] = df[content_features].mean(axis=1)
    df['Content_features_std'] = df[content_features].std(axis=1)
    
    return df


def process_dataset(input_path: Path, output_path: Path, dataset_name: str) -> None:
    """Process a single dataset (train or test) with feature engineering."""
    logger.info(f"Processing {dataset_name} data from {input_path}")
    
    # Load the raw data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns for {dataset_name}")
    
    # Store original columns for reference
    original_columns = df.columns.tolist()
    
    # Apply all feature engineering steps
    df = create_ratio_features(df)
    df = create_multiplicative_features(df)
    df = create_nonlinear_transforms(df)
    df = create_cyclical_features(df)
    df = create_difference_features(df)
    df = create_duration_normalizations(df)
    df = create_polynomial_features(df)
    df = create_statistical_features(df)
    
    # Log feature creation summary
    new_features = [col for col in df.columns if col not in original_columns]
    logger.info(f"Created {len(new_features)} new features for {dataset_name}")
    logger.info(f"Total features for {dataset_name}: {len(df.columns)}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the enhanced dataset
    logger.info(f"Saving {dataset_name} features to {output_path}")
    df.to_csv(output_path, index=False)
    
    logger.success(f"{dataset_name.capitalize()} feature engineering complete! Created {len(new_features)} new features.")
    logger.info(f"Final {dataset_name} dataset shape: {df.shape}")


@app.command()
def main(
    train_input: Path = RAW_DATA_DIR / "train.csv",
    test_input: Path = RAW_DATA_DIR / "test.csv",
    train_output: Path = PROCESSED_DATA_DIR / "train_features.csv",
    test_output: Path = PROCESSED_DATA_DIR / "test_features.csv",
):
    """Generate comprehensive features from both train and test datasets."""
    logger.info("Starting feature engineering for train and test datasets...")
    
    # Process training data
    process_dataset(train_input, train_output, "train")
    
    # Process test data
    process_dataset(test_input, test_output, "test")
    
    logger.success("Feature engineering complete for both train and test datasets!")


if __name__ == "__main__":
    app()
