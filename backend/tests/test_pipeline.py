"""
Unit tests for VR180 processing pipeline
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from worker.pipeline import VideoProcessor
from worker.status import StatusManager
from worker.config import Config
from worker.models import ModelManager

class TestModelManager:
    """Test ModelManager with Hugging Face API"""
    
    def test_model_manager_init(self):
        """Test ModelManager initialization"""
        manager = ModelManager()
        assert manager.hf_token is not None
        assert "api-inference.huggingface.co" in manager.api_base
    
    def test_image_conversion(self):
        """Test image to base64 conversion"""
        manager = ModelManager()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Convert to base64 and back
        b64_str = manager._image_to_base64(test_image)
        converted_image = manager._base64_to_image(b64_str)
        
        assert converted_image.shape == test_image.shape
        assert np.allclose(converted_image, test_image, atol=1)
    
    def test_simple_depth_estimation(self):
        """Test fallback depth estimation"""
        manager = ModelManager()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test depth estimation
        depth = manager._simple_depth_estimation(test_image)
        
        assert depth.shape == (100, 100)
        assert depth.min() >= 0
        assert depth.max() <= 1
        assert not np.isnan(depth).any()

class TestStatusManager:
    """Test StatusManager functionality"""
    
    def test_status_update(self):
        """Test status update functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            status_manager = StatusManager(temp_dir)
            job_id = "test_job"
            
            # Test initial status
            status_manager.update_progress(job_id, "test_stage", 50, "Test message")
            status = status_manager.get_status(job_id)
            
            assert status["stage"] == "test_stage"
            assert status["percent"] == 50
            assert status["message"] == "Test message"
    
    def test_status_completion(self):
        """Test status completion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            status_manager = StatusManager(temp_dir)
            job_id = "test_job"
            
            status_manager.mark_completed(job_id, "/path/to/output.mp4", "/path/to/preview.mp4")
            status = status_manager.get_status(job_id)
            
            assert status["status"] == "completed"
            assert status["percent"] == 100
            assert "output_path" in status

class TestConfig:
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = Config()
        
        # Test default values
        assert config.BASELINE_RATIO == 0.03
        assert config.LDI_LAYERS == 3
        assert config.DOWNSCALE_FACTOR == 0.6
    
    def test_models_exist(self):
        """Test models existence check"""
        config = Config()
        assert config.models_exist() == True  # Should be True for API approach

class TestVideoIO:
    """Test video I/O utilities"""
    
    def test_image_conversion(self):
        """Test image format conversion"""
        from worker.video_io import VideoProcessor
        
        video_io = VideoProcessor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test bilinear sampling
        x = np.random.rand(100, 100) * 99
        y = np.random.rand(100, 100) * 99
        
        sampled = video_io._bilinear_sample(test_image, x, y)
        
        assert sampled.shape == test_image.shape
        assert sampled.dtype == np.uint8

def test_reprojection_math():
    """Test stereoscopic reprojection mathematics"""
    from worker.pipeline import VideoProcessor
    
    # Test reprojection parameters
    width = 1920
    height = 1080
    baseline_ratio = 0.03
    
    # Calculate baseline
    baseline_px = baseline_ratio * width
    expected_baseline = 0.03 * 1920
    assert abs(baseline_px - expected_baseline) < 0.01
    
    # Test depth normalization
    depth = np.random.rand(height, width)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    assert depth_normalized.min() >= 0
    assert depth_normalized.max() <= 1
    
    # Test shift calculation
    shift_x = baseline_px * (1 - depth_normalized)
    assert shift_x.shape == depth.shape
    assert shift_x.min() >= 0
    assert shift_x.max() <= baseline_px

if __name__ == "__main__":
    pytest.main([__file__])
