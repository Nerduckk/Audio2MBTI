"""
Schema Versioning Module
Manages CSV schema versions and migrations
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
from pathlib import Path
from .config_loader import ConfigLoader, get_logger

logger = get_logger(__name__)


class SchemaVersion:
    """Represents a CSV schema version"""
    
    def __init__(self, version: str, columns: List[str], description: str = ""):
        self.version = version
        self.columns = columns
        self.description = description
        self.created_at = datetime.now().isoformat()
        self.schema_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash of schema"""
        schema_str = json.dumps(self.columns, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "columns": self.columns,
            "description": self.description,
            "created_at": self.created_at,
            "schema_hash": self.schema_hash,
            "column_count": len(self.columns)
        }


class SchemaVersionControl:
    """Manages schema versions and migrations"""
    
    # Current schema version
    CURRENT_VERSION = "1.0"
    
    # Schema versions history
    SCHEMAS = {
        "1.0": SchemaVersion(
            "1.0",
            [
                "title", "artists", "album", "release_date", "genre",
                "tempo", "tempo_strength", "energy", "danceability", "valence",
                "acousticness", "instrumentalness", "liveness", "speechiness", "loudness",
                "zero_crossing_rate", "spectral_bandwidth", "spectral_centroid",
                "mfcc_mean", "mfcc_std",
                "lyrics", "lyrics_wordcount", "lyrics_polarity",
                "emotion_class", "emotion_confidence",
                "genre_ei", "genre_sn", "genre_tf", "mbti_label"
            ],
            "Initial schema with audio features, lyrics sentiment, and MBTI labels"
        )
    }
    
    def __init__(self):
        self.current_schema = self.SCHEMAS[self.CURRENT_VERSION]
        self._load_version_history()
    
    def _load_version_history(self):
        """Load version history from disk if exists"""
        version_file = "schema_version_history.json"
        self.version_history = {}
        
        if Path(version_file).exists():
            try:
                with open(version_file, 'r') as f:
                    self.version_history = json.load(f)
                logger.debug(f"Loaded schema version history: {len(self.version_history)} versions")
            except Exception as e:
                logger.warning(f"Could not load version history: {e}")
    
    def get_current_schema(self) -> SchemaVersion:
        """Get current schema version"""
        return self.current_schema
    
    def validate_dataframe_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame matches current schema
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result dictionary
        """
        expected_columns = set(self.current_schema.columns)
        actual_columns = set(df.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        is_valid = len(missing_columns) == 0 and len(extra_columns) == 0
        
        result = {
            "is_valid": is_valid,
            "schema_version": self.CURRENT_VERSION,
            "expected_columns": len(expected_columns),
            "actual_columns": len(actual_columns),
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "column_order_matches": list(df.columns) == self.current_schema.columns
        }
        
        if not is_valid:
            logger.warning(f"Schema validation failed: {result}")
        else:
            logger.debug("Schema validation passed")
        
        return result
    
    def add_schema_version(self, version: str, columns: List[str], 
                          description: str = "") -> SchemaVersion:
        """Add new schema version"""
        schema = SchemaVersion(version, columns, description)
        self.SCHEMAS[version] = schema
        
        logger.info(f"Added schema version {version}: {len(columns)} columns")
        
        self._save_version_history()
        return schema
    
    def create_migration_report(self, old_df: pd.DataFrame, 
                               new_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create migration report between two DataFrames
        
        Args:
            old_df: Old DataFrame
            new_df: New DataFrame
            
        Returns:
            Migration report
        """
        return {
            "old_rows": len(old_df),
            "new_rows": len(new_df),
            "old_columns": len(old_df.columns),
            "new_columns": len(new_df.columns),
            "rows_added": len(new_df) - len(old_df),
            "new_columns": list(set(new_df.columns) - set(old_df.columns)),
            "removed_columns": list(set(old_df.columns) - set(new_df.columns)),
            "migration_timestamp": datetime.now().isoformat()
        }
    
    def _save_version_history(self):
        """Save version history to disk"""
        try:
            history = {
                v: schema.to_dict() for v, schema in self.SCHEMAS.items()
            }
            with open("schema_version_history.json", 'w') as f:
                json.dump(history, f, indent=2)
            logger.debug("Saved schema version history")
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")


class CSVMigrationManager:
    """Manages CSV migrations with backup versioning"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.schema_vc = SchemaVersionControl()
        self.backup_dir = ConfigLoader.get("paths.csv_backup_dir", "./data/backups")
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
    
    def backup_current(self, reason: str = "") -> str:
        """
        Create backup of current CSV with version info
        
        Args:
            reason: Reason for backup
            
        Returns:
            Path to backup file
        """
        if not Path(self.csv_file).exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(self.backup_dir) / f"{Path(self.csv_file).stem}_v{self.schema_vc.CURRENT_VERSION}_{timestamp}.csv"
        
        try:
            df = pd.read_csv(self.csv_file)
            df.to_csv(backup_file, index=False)
            
            logger.info(f"Backed up {self.csv_file} to {backup_file}")
            
            # Log backup info
            backup_info = {
                "original_file": self.csv_file,
                "backup_file": str(backup_file),
                "schema_version": self.schema_vc.CURRENT_VERSION,
                "rows": len(df),
                "columns": len(df.columns),
                "timestamp": datetime.now().isoformat(),
                "reason": reason
            }
            
            self._save_backup_manifest(backup_info)
            
            return str(backup_file)
        
        except Exception as e:
            logger.error(f"Failed to backup CSV: {e}")
            return None
    
    def validate_csv(self) -> Dict[str, Any]:
        """Validate CSV matches current schema"""
        if not Path(self.csv_file).exists():
            return {"exists": False, "is_valid": False}
        
        try:
            df = pd.read_csv(self.csv_file)
            validation = self.schema_vc.validate_dataframe_schema(df)
            validation["exists"] = True
            validation["file_size"] = Path(self.csv_file).stat().st_size
            validation["rows"] = len(df)
            return validation
        except Exception as e:
            logger.error(f"Failed to validate CSV: {e}")
            return {"exists": True, "is_valid": False, "error": str(e)}
    
    def _save_backup_manifest(self, backup_info: Dict[str, Any]):
        """Save backup information to manifest"""
        manifest_file = Path(self.backup_dir) / "backup_manifest.json"
        
        try:
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
            else:
                manifest = []
            
            manifest.append(backup_info)
            
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save backup manifest: {e}")
