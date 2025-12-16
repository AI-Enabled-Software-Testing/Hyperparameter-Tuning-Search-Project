"""Pytest tests for analyze_experiment.py and refactored modules."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils import _exp_smooth
from constants import (
    SMOOTHING_SLIDER_BEST_FITNESS,
    SMOOTHING_SLIDER_CURRENT_FITNESS,
    SMOOTHING_SLIDER_INDIVIDUAL_METRICS,
    METRIC_WEIGHTS,
)
from analyze_experiment import (
    load_experiment_runs,
    extract_metric_convergence,
    diagnose_pso_hyperparameters,
)


class TestUtils:
    """Test utility functions."""
    
    def test_exp_smooth_basic(self):
        """Test exponential smoothing with simple data."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = _exp_smooth(values, smoothing_slider=0.5)
        
        assert len(smoothed) == len(values)
        assert smoothed[0] == values[0]  # First value unchanged
        assert isinstance(smoothed, np.ndarray)
        assert smoothed.dtype == np.float32
    
    def test_exp_smooth_high_smoothing(self):
        """Test with high smoothing factor (more smoothing)."""
        values = np.array([1.0, 10.0, 1.0, 10.0])
        smoothed = _exp_smooth(values, smoothing_slider=0.9)
        
        # With high smoothing, changes should be dampened
        assert abs(smoothed[1] - smoothed[0]) < abs(values[1] - values[0])
    
    def test_exp_smooth_low_smoothing(self):
        """Test with low smoothing factor (less smoothing)."""
        values = np.array([1.0, 10.0, 1.0, 10.0])
        smoothed = _exp_smooth(values, smoothing_slider=0.1)
        
        # With low smoothing, should be closer to original
        assert abs(smoothed[1] - values[1]) < 2.0


class TestConstants:
    """Test constants module."""
    
    def test_smoothing_sliders_exist(self):
        """Test that all smoothing slider constants exist."""
        assert isinstance(SMOOTHING_SLIDER_BEST_FITNESS, float)
        assert isinstance(SMOOTHING_SLIDER_CURRENT_FITNESS, float)
        assert isinstance(SMOOTHING_SLIDER_INDIVIDUAL_METRICS, float)
    
    def test_smoothing_sliders_valid_range(self):
        """Test that smoothing sliders are in valid range [0, 1]."""
        assert 0 <= SMOOTHING_SLIDER_BEST_FITNESS <= 1
        assert 0 <= SMOOTHING_SLIDER_CURRENT_FITNESS <= 1
        assert 0 <= SMOOTHING_SLIDER_INDIVIDUAL_METRICS <= 1
    
    def test_metric_weights_structure(self):
        """Test metric weights dictionary structure."""
        assert isinstance(METRIC_WEIGHTS, dict)
        assert len(METRIC_WEIGHTS) == 6
        
        expected_metrics = [
            "f1_macro", "recall_macro", "roc_auc",
            "precision_macro", "accuracy", "f1_micro"
        ]
        for metric in expected_metrics:
            assert metric in METRIC_WEIGHTS
            assert isinstance(METRIC_WEIGHTS[metric], float)
    
    def test_metric_weights_sum(self):
        """Test that metric weights sum to 1.0."""
        total = sum(METRIC_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01  # Allow small floating point error


class TestLoadExperimentRuns:
    """Test load_experiment_runs function."""
    
    @pytest.fixture
    def mock_experiment_data(self):
        """Create mock experiment data."""
        return {
            "summary": {
                "final_fitness": 0.85,
                "total_time_sec": 120.5,
                "best_params": {"lr": 0.01}
            },
            "convergence": {
                "evaluations": [1, 2, 3, 4, 5],
                "best_fitness": [0.5, 0.6, 0.7, 0.75, 0.8],
                "current_fitness": [0.5, 0.55, 0.65, 0.7, 0.75]
            },
            "history": [
                {
                    "trial": 1,
                    "metrics": {
                        "f1_macro": 0.5, "accuracy": 0.6,
                        "recall_macro": 0.55, "precision_macro": 0.52,
                        "roc_auc": 0.58, "f1_micro": 0.51
                    }
                },
                {
                    "trial": 2,
                    "metrics": {
                        "f1_macro": 0.6, "accuracy": 0.65,
                        "recall_macro": 0.62, "precision_macro": 0.58,
                        "roc_auc": 0.63, "f1_micro": 0.61
                    }
                }
            ]
        }
    
    def test_load_experiment_runs_convergence(self, tmp_path, mock_experiment_data):
        """Test loading experiment runs with convergence data."""
        # Create mock directory structure
        experiment_dir = tmp_path / "test-experiment"
        run_dir = experiment_dir / "run_001"
        run_dir.mkdir(parents=True)
        
        # Write mock data
        with open(run_dir / "summary.json", "w") as f:
            json.dump(mock_experiment_data["summary"], f)
        
        with open(run_dir / "convergence.json", "w") as f:
            json.dump(mock_experiment_data["convergence"], f)
        
        # Patch EXPERIMENT_ROOT
        with patch("analyze_experiment.EXPERIMENT_ROOT", tmp_path):
            runs = load_experiment_runs("test-experiment", include_history=False)
        
        assert len(runs) == 1
        assert "convergence" in runs[0]
        assert "summary" in runs[0]
        assert runs[0]["run_dir"] == "run_001"
        assert runs[0]["summary"]["final_fitness"] == 0.85
    
    def test_load_experiment_runs_history(self, tmp_path, mock_experiment_data):
        """Test loading experiment runs with history data."""
        # Create mock directory structure
        experiment_dir = tmp_path / "test-experiment"
        run_dir = experiment_dir / "run_001"
        run_dir.mkdir(parents=True)
        
        # Write mock data
        with open(run_dir / "summary.json", "w") as f:
            json.dump(mock_experiment_data["summary"], f)
        
        with open(run_dir / "history.json", "w") as f:
            json.dump(mock_experiment_data["history"], f)
        
        # Patch EXPERIMENT_ROOT
        with patch("analyze_experiment.EXPERIMENT_ROOT", tmp_path):
            runs = load_experiment_runs("test-experiment", include_history=True)
        
        assert len(runs) == 1
        assert "history" in runs[0]
        assert "summary" in runs[0]
        assert len(runs[0]["history"]) == 2
    
    def test_load_experiment_runs_missing_experiment(self, tmp_path):
        """Test error handling for missing experiment."""
        with patch("analyze_experiment.EXPERIMENT_ROOT", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_experiment_runs("nonexistent-experiment")
    
    def test_load_experiment_runs_no_valid_runs(self, tmp_path):
        """Test error handling when no valid runs exist."""
        experiment_dir = tmp_path / "test-experiment"
        experiment_dir.mkdir(parents=True)
        
        # Create incomplete run (missing files)
        run_dir = experiment_dir / "run_001"
        run_dir.mkdir()
        
        with patch("analyze_experiment.EXPERIMENT_ROOT", tmp_path):
            with pytest.raises(ValueError, match="No valid runs found"):
                load_experiment_runs("test-experiment")


class TestExtractMetricConvergence:
    """Test extract_metric_convergence function."""
    
    def test_extract_metric_convergence_basic(self):
        """Test extracting convergence for a single metric."""
        history = [
            {"trial": 1, "metrics": {"f1_macro": 0.5, "accuracy": 0.6}},
            {"trial": 2, "metrics": {"f1_macro": 0.7, "accuracy": 0.65}},
            {"trial": 3, "metrics": {"f1_macro": 0.6, "accuracy": 0.7}},
            {"trial": 4, "metrics": {"f1_macro": 0.8, "accuracy": 0.75}},
        ]
        
        result = extract_metric_convergence(history, "f1_macro")
        
        assert "evaluations" in result
        assert "best_metric" in result
        assert "current_metric" in result
        
        assert result["evaluations"] == [1, 2, 3, 4]
        assert result["current_metric"] == [0.5, 0.7, 0.6, 0.8]
        # Best should be monotonically increasing
        assert result["best_metric"] == [0.5, 0.7, 0.7, 0.8]
    
    def test_extract_metric_convergence_monotonic(self):
        """Test that best_metric is monotonically increasing."""
        history = [
            {"trial": 1, "metrics": {"accuracy": 0.5}},
            {"trial": 2, "metrics": {"accuracy": 0.3}},  # Worse
            {"trial": 3, "metrics": {"accuracy": 0.7}},  # Better
            {"trial": 4, "metrics": {"accuracy": 0.6}},  # Worse
        ]
        
        result = extract_metric_convergence(history, "accuracy")
        best_vals = result["best_metric"]
        
        # Check monotonically increasing
        for i in range(1, len(best_vals)):
            assert best_vals[i] >= best_vals[i-1]


class TestDiagnosePSOHyperparameters:
    """Test PSO hyperparameter diagnostics."""
    
    @pytest.fixture
    def mock_pso_runs(self):
        """Create mock PSO runs with various patterns."""
        # Run with premature convergence
        premature_run = {
            "convergence": {
                "evaluations": list(range(100)),
                "best_fitness": [0.5] + [0.8] * 99,  # Plateaus early
                "current_fitness": [0.5 + 0.003 * i for i in range(100)]
            }
        }
        
        # Run with slow convergence
        slow_run = {
            "convergence": {
                "evaluations": list(range(100)),
                "best_fitness": [0.5 + 0.003 * i for i in range(100)],  # Continuous improvement
                "current_fitness": [0.5 + 0.003 * i for i in range(100)]
            }
        }
        
        return [premature_run, slow_run]
    
    def test_diagnose_pso_hyperparameters_runs(self, mock_pso_runs, capsys):
        """Test PSO diagnostics with mock runs."""
        diagnose_pso_hyperparameters(mock_pso_runs, "test-pso")
        
        captured = capsys.readouterr()
        
        # Should print diagnostic header
        assert "PSO HYPERPARAMETER DIAGNOSTIC" in captured.out
        
        # Should analyze both runs
        assert "Run 1" in captured.out
        assert "Run 2" in captured.out
        
        # Should show improvement metrics
        assert "Total improvement" in captured.out
    
    def test_diagnose_pso_non_pso_experiment(self, mock_pso_runs, capsys):
        """Test that non-PSO experiments show a note."""
        diagnose_pso_hyperparameters(mock_pso_runs, "test-ga")
        
        captured = capsys.readouterr()
        assert "PSO experiments only" in captured.out


class TestImports:
    """Test that all imports work correctly."""
    
    def test_import_utils(self):
        """Test importing utils module."""
        from utils import _exp_smooth
        assert callable(_exp_smooth)
    
    def test_import_constants(self):
        """Test importing constants module."""
        from constants import (
            SMOOTHING_SLIDER_BEST_FITNESS,
            SMOOTHING_SLIDER_CURRENT_FITNESS,
            METRIC_WEIGHTS,
        )
        assert isinstance(METRIC_WEIGHTS, dict)
    
    def test_import_plots(self):
        """Test importing plots module."""
        from plots import (
            plot_best_fitness_overlay,
            plot_current_fitness_overlay,
            plot_convergence,
        )
        assert callable(plot_best_fitness_overlay)
        assert callable(plot_current_fitness_overlay)
        assert callable(plot_convergence)
    
    def test_import_analyze_experiment(self):
        """Test importing analyze_experiment module."""
        from analyze_experiment import (
            load_experiment_runs,
            extract_metric_convergence,
        )
        assert callable(load_experiment_runs)
        assert callable(extract_metric_convergence)


class TestPlotsIntegration:
    """Test that plots can use refactored modules."""
    
    def test_plots_use_constants(self):
        """Test that plots module correctly uses constants."""
        import plots
        
        # Plots should have access to these constants
        assert hasattr(plots, 'METRIC_WEIGHTS')
        assert hasattr(plots, 'SMOOTHING_SLIDER_BEST_FITNESS')
        
        # Should be the same objects
        assert plots.METRIC_WEIGHTS is METRIC_WEIGHTS
    
    def test_plots_use_utils(self):
        """Test that plots module correctly uses utils."""
        import plots
        
        # Plots should have access to _exp_smooth
        assert hasattr(plots, '_exp_smooth')
        
        # Test that it works
        test_data = np.array([1.0, 2.0, 3.0])
        smoothed = plots._exp_smooth(test_data, 0.5)
        assert len(smoothed) == len(test_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
