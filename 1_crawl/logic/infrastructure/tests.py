"""
Unit Tests and Integration Tests for Audio2MBTI
Run with: python -m infrastructure.tests
"""

import unittest
import pandas as pd
import numpy as np
import sys

from infrastructure.batch_processor import BatchProcessor
from infrastructure.data_validator import DataValidator
from infrastructure.retry_logic import RateLimiter, RetryConfig
from infrastructure.schema_versioning import SchemaVersionControl


class TestDataValidator(unittest.TestCase):
    """Unit tests for data validation"""
    
    def setUp(self):
        self.validator = DataValidator()
    
    def test_validate_audio_features_valid(self):
        """Test validation with valid audio features"""
        features = {
            "tempo_bpm": 120,
            "energy": 0.7,
            "danceability": 0.65,
            "spectral_centroid": 2500,
            "spectral_flatness": 0.3,
            "zero_crossing_rate": 0.08,
            "spectral_bandwidth": 0.4,
            "spectral_rolloff": 0.5,
            "tempo_strength": 0.4,
        }
        
        is_valid, errors = self.validator.validate_audio_features(features)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_audio_features_invalid_range(self):
        """Test validation with out-of-range values"""
        features = {
            "tempo_bpm": 400,  # Out of range
            "energy": 0.7,
            "danceability": 0.65,
            "spectral_centroid": 2500,
            "spectral_flatness": 0.3,
            "zero_crossing_rate": 0.08,
            "spectral_bandwidth": 0.4,
            "spectral_rolloff": 0.5,
            "tempo_strength": 0.4,
        }
        
        is_valid, errors = self.validator.validate_audio_features(features)
        self.assertFalse(is_valid)
        self.assertTrue(any("tempo_bpm" in e for e in errors))
    
    def test_validate_audio_features_nan(self):
        """Test validation with NaN values"""
        features = {
            "tempo_bpm": np.nan,
            "energy": 0.7,
            "danceability": 0.65,
            "spectral_centroid": 2500,
            "spectral_flatness": 0.3,
            "zero_crossing_rate": 0.08,
            "spectral_bandwidth": 0.4,
            "spectral_rolloff": 0.5,
            "tempo_strength": 0.4,
        }
        
        is_valid, errors = self.validator.validate_audio_features(features)
        self.assertFalse(is_valid)
        self.assertTrue(any("NaN" in e for e in errors))
    
    def test_validate_row_missing_required_column(self):
        """Test row validation with missing required column"""
        row = {
            "title": "Test Song",
            # Missing artists
            "artist_genres": "pop",
            "tempo_bpm": 120
        }
        
        is_valid, errors = self.validator.validate_row(
            row,
            required_columns=["title", "artists", "artist_genres"]
        )
        
        self.assertFalse(is_valid)
        self.assertTrue(any("artists" in e for e in errors))
    
    def test_validate_lyrics_data_valid(self):
        """Test lyrics validation with valid data"""
        lyrics = {
            "polarity": 0.5,
            "wordcount": 100
        }
        
        is_valid, errors = self.validator.validate_lyrics_data(lyrics)
        self.assertTrue(is_valid)
    
    def test_validate_lyrics_data_invalid_polarity(self):
        """Test lyrics validation with invalid polarity"""
        lyrics = {
            "polarity": 2.0,  # Out of range
            "wordcount": 100
        }
        
        is_valid, errors = self.validator.validate_lyrics_data(lyrics)
        self.assertFalse(is_valid)


class TestRetryLogic(unittest.TestCase):
    """Unit tests for retry logic"""
    
    def test_retry_config_exponential_backoff(self):
        """Test exponential backoff calculation"""
        config = RetryConfig(
            max_retries=5,
            initial_delay=1.0,
            max_delay=300,
            backoff_multiplier=2.0,
            jitter=False,
            strategy="exponential_backoff"
        )
        
        delays = [config.get_delay(i) for i in range(5)]
        
        # Should double each time: 1, 2, 4, 8, 16
        expected = [1.0, 2.0, 4.0, 8.0, 16.0]
        for actual, exp in zip(delays, expected):
            self.assertAlmostEqual(actual, exp, places=1)
    
    def test_retry_config_linear_backoff(self):
        """Test linear backoff calculation"""
        config = RetryConfig(
            max_retries=5,
            initial_delay=1.0,
            backoff_multiplier=1.0,
            jitter=False,
            strategy="linear_backoff"
        )
        
        delays = [config.get_delay(i) for i in range(5)]
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for actual, exp in zip(delays, expected):
            self.assertAlmostEqual(actual, exp, places=1)
    
    def test_rate_limiter(self):
        """Test rate limiter spacing"""
        limiter = RateLimiter(requests_per_second=2.0)  # 0.5s per request
        
        import time
        start = time.time()
        for _ in range(3):
            limiter.wait()
        elapsed = time.time() - start
        
        # Should take at least ~1 second for 3 requests at 2/sec
        self.assertGreater(elapsed, 0.9)


class TestBatchProcessor(unittest.TestCase):
    """Unit tests for batch processing"""
    
    def setUp(self):
        self.temp_file = "test_batch_output.csv"
    
    def tearDown(self):
        import os
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_batch_processor_accumulation(self):
        """Test batch accumulation before flush"""
        processor = BatchProcessor(batch_size=3, output_file=self.temp_file)
        
        # Add records below batch_size
        processor.add({"id": 1, "value": "a"})
        processor.add({"id": 2, "value": "b"})
        
        # Batch should not be flushed yet
        self.assertEqual(len(processor.batch), 2)
        
        # This should trigger flush
        processor.add({"id": 3, "value": "c"})
        
        # After reaching batch_size, batch should be flushed
        self.assertEqual(len(processor.batch), 0)
        self.assertEqual(processor.total_saved, 3)
    
    def test_batch_processor_manual_flush(self):
        """Test manual batch flush"""
        processor = BatchProcessor(batch_size=10, output_file=self.temp_file)
        
        processor.add({"id": 1, "value": "test"})
        records_written = processor.flush()
        
        self.assertEqual(records_written, 1)
        self.assertEqual(len(processor.batch), 0)


class TestSchemaVersioning(unittest.TestCase):
    """Unit tests for schema versioning"""
    
    def setUp(self):
        self.schema_vc = SchemaVersionControl()
    
    def test_schema_version_creation(self):
        """Test schema version creation"""
        schema = self.schema_vc.get_current_schema()
        
        self.assertIsNotNone(schema)
        self.assertEqual(schema.version, "1.0")
        self.assertGreater(len(schema.columns), 0)
    
    def test_validate_dataframe_schema_valid(self):
        """Test validation with matching schema"""
        schema = self.schema_vc.get_current_schema()
        df = pd.DataFrame(columns=schema.columns)
        
        # Add one row of valid data
        df.loc[0] = [None] * len(schema.columns)
        
        validation = self.schema_vc.validate_dataframe_schema(df)
        
        self.assertTrue(validation["is_valid"])
        self.assertEqual(len(validation["missing_columns"]), 0)
        self.assertEqual(len(validation["extra_columns"]), 0)
    
    def test_validate_dataframe_schema_missing_columns(self):
        """Test validation with missing columns"""
        schema = self.schema_vc.get_current_schema()
        df = pd.DataFrame(columns=schema.columns[:-3])  # Remove last 3 columns
        
        validation = self.schema_vc.validate_dataframe_schema(df)
        
        self.assertFalse(validation["is_valid"])
        self.assertEqual(len(validation["missing_columns"]), 3)


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def test_complete_data_pipeline(self):
        """Test complete data validation and batching pipeline"""
        validator = DataValidator()
        processor = BatchProcessor(batch_size=2)
        
        # Create valid records
        records = [
            {
                "title": "Song 1",
                "artists": "Artist 1",
                "artist_genres": "pop",
                "tempo_bpm": 120,
                "energy": 0.7,
                "danceability": 0.6,
                "spectral_centroid": 2500,
                "lyrics_polarity": 0.5,
                "mbti_label": "ENFP"
            },
            {
                "title": "Song 2",
                "artists": "Artist 2",
                "artist_genres": "rock",
                "tempo_bpm": 140,
                "energy": 0.8,
                "danceability": 0.5,
                "spectral_centroid": 3000,
                "lyrics_polarity": 0.6,
                "mbti_label": "ISTJ"
            }
        ]
        
        valid_count = 0
        for record in records:
            # Validate before processing
            required_cols = ["title", "artists", "artist_genres", "mbti_label"]
            is_valid, errors = validator.validate_row(record, required_cols)
            
            if is_valid:
                valid_count += 1
        
        self.assertEqual(valid_count, 2)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestRetryLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaVersioning))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
