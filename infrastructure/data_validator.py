"""
Data Validation Module
Validates input data, audio features, and output data quality
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config_loader import ConfigLoader, get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates data at various pipeline stages"""
    
    def __init__(self):
        self.config = ConfigLoader.get("validation", {})
        self.quality_metrics = {
            "records_processed": 0,
            "records_valid": 0,
            "records_invalid": 0,
            "validation_errors": {},
            "warnings": [],
            "timestamp": None
        }
    
    def validate_audio_features(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate extracted audio features are within acceptable ranges
        
        Args:
            features: Dictionary of audio features to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        ranges = self.config.get("audio_features", {})
        
        validation_rules = {
            "tempo_bpm": ranges.get("tempo_range", [60, 240]),
            "energy": ranges.get("energy_range", [0, 1]),
            "danceability": ranges.get("danceability_range", [0, 1]),
            "spectral_centroid": [0, 10000],
            "spectral_flatness": [0, 1],
            "zero_crossing_rate": [0, 1],
            "spectral_bandwidth": [0, 1],
            "spectral_rolloff": [0, 1],
            "tempo_strength": [0, 1],
        }
        
        for feature, valid_range in validation_rules.items():
            if feature not in features:
                errors.append(f"Missing required feature: {feature}")
                continue
            
            value = features[feature]
            
            # Check for NaN/ None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                errors.append(f"Feature '{feature}' is NaN or None")
                continue
            
            # Check range
            try:
                value = float(value)
                min_val, max_val = valid_range
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"Feature '{feature}' ({value}) outside range [{min_val}, {max_val}]"
                    )
            except (ValueError, TypeError) as e:
                errors.append(f"Feature '{feature}' invalid type: {e}")
        
        return len(errors) == 0, errors
    
    def validate_lyrics_data(self, lyrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate lyrics analysis results
        
        Args:
            lyrics: Dictionary with polarity, wordcount, emotion_class, etc.
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        ranges = self.config.get("lyrics", {})
        
        # Validate polarity
        if "polarity" in lyrics:
            polarity = lyrics["polarity"]
            if polarity is None or (isinstance(polarity, float) and np.isnan(polarity)):
                errors.append("Lyrics polarity is NaN")
            else:
                try:
                    polarity = float(polarity)
                    min_val, max_val = ranges.get("polarity_range", [-1, 1])
                    if not (min_val <= polarity <= max_val):
                        errors.append(f"Polarity {polarity} outside range [{min_val}, {max_val}]")
                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid polarity type: {e}")
        
        # Validate wordcount
        if "wordcount" in lyrics:
            wordcount = lyrics["wordcount"]
            if wordcount is not None:
                try:
                    wordcount = int(wordcount)
                    min_words = ranges.get("min_words", 5)
                    max_words = ranges.get("max_words", 10000)
                    if wordcount > 0 and (wordcount < min_words or wordcount > max_words):
                        logger.warning(
                            f"Wordcount {wordcount} outside typical range [{min_words}, {max_words}]"
                        )
                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid wordcount type: {e}")
        
        return len(errors) == 0, errors
    
    def validate_row(self, row: Dict[str, Any], required_columns: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate a complete data row
        
        Args:
            row: Dictionary representing one data row
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if required_columns is None:
            required_columns = self.config.get("required_columns", [])
        
        # Check required columns
        for col in required_columns:
            if col not in row or row[col] is None or row[col] == "":
                errors.append(f"Missing required column: {col}")
        
        # Check for NaN in critical columns
        for col in required_columns:
            if col in row:
                value = row[col]
                if isinstance(value, float) and np.isnan(value):
                    errors.append(f"NaN value in required column: {col}")
        
        # Validate specific fields if present
        if "tempo_bpm" in row and row["tempo_bpm"] is not None:
            is_valid, feature_errors = self.validate_audio_features({
                "tempo_bpm": row.get("tempo_bpm"),
                "energy": row.get("energy", 0.5),
                "danceability": row.get("danceability", 0.5),
                "spectral_centroid": row.get("spectral_centroid", 0.0),
                "spectral_flatness": row.get("spectral_flatness", 0.0),
                "zero_crossing_rate": row.get("zero_crossing_rate", 0.0),
                "spectral_bandwidth": row.get("spectral_bandwidth", 0.0),
                "spectral_rolloff": row.get("spectral_rolloff", 0.0),
                "tempo_strength": row.get("tempo_strength", 0.0),
            })
            if not is_valid:
                errors.extend(feature_errors)
        
        if "lyrics_polarity" in row and row["lyrics_polarity"] is not None:
            is_valid, lyrics_errors = self.validate_lyrics_data({
                "polarity": row.get("lyrics_polarity")
            })
            if not is_valid:
                errors.extend(lyrics_errors)
        
        return len(errors) == 0, errors
    
    def validate_dataframe(self, df: pd.DataFrame, check_null: bool = True) -> Dict[str, Any]:
        """
        Validate entire DataFrame and generate quality report
        
        Args:
            df: DataFrame to validate
            check_null: Whether to check for null values
            
        Returns:
            Dictionary with validation results and metrics
        """
        required_columns = self.config.get("required_columns", [])
        
        missing_columns = set(required_columns) - set(df.columns)
        null_counts = df.isnull().sum() if check_null else {}
        null_pct = (null_counts / len(df) * 100).round(2) if check_null else {}
        
        validation_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_columns": list(missing_columns),
            "null_counts": null_counts.to_dict() if check_null else {},
            "null_percentage": null_pct.to_dict() if check_null else {},
            "data_types": df.dtypes.to_dict(),
            "invalid_rows": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Validate each row
        for idx, row in df.iterrows():
            is_valid, errors = self.validate_row(row.to_dict(), required_columns)
            if not is_valid:
                validation_report["invalid_rows"].append({
                    "row_index": idx,
                    "errors": errors
                })
        
        validation_report["valid_rows"] = len(df) - len(validation_report["invalid_rows"])
        validation_report["is_valid"] = len(missing_columns) == 0 and \
                                       len(validation_report["invalid_rows"]) == 0
        
        # Log summary
        logger.info(f"DataFrame validation: {validation_report['valid_rows']}/{len(df)} rows valid")
        
        return validation_report
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality check
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "duplicates": len(df[df.duplicated()]),
            "columns": {}
        }
        
        # Check each numeric column
        for col in df.select_dtypes(include=[np.number]).columns:
            quality_report["columns"][col] = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "null_percent": round(df[col].isnull().sum() / len(df) * 100, 2),
                "min": float(df[col].min()) if len(df[col]) > 0 else None,
                "max": float(df[col].max()) if len(df[col]) > 0 else None,
                "mean": float(df[col].mean()) if len(df[col]) > 0 else None,
                "std": float(df[col].std()) if len(df[col]) > 0 else None,
                "outliers": int((np.abs(df[col] - df[col].mean()) > 3 * df[col].std()).sum())
            }
        
        logger.info(f"Data quality check: {quality_report['total_rows']} rows, "
                   f"{quality_report['duplicates']} duplicates")
        
        return quality_report
    
    def log_validation_report(self, validation_report: Dict[str, Any], filename: str = None):
        """Save validation report to JSON file"""
        if filename is None:
            filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs("logs", exist_ok=True)
        filepath = f"logs/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Validation report saved to {filepath}")
