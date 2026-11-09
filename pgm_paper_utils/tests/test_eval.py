"""Tests for the evaluation module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from jax import random

from pgm_paper_utils.eval import evaluate_distribution


class TestEvaluateDistributionV3:
    """Test cases for the main evaluation function."""
    
    @pytest.fixture
    def mock_pgm(self):
        """Create a mock PGM object."""
        pgm = Mock()
        pgm.sample = Mock(return_value={
            'x': jnp.ones((10,)),
            'y': jnp.ones((10,))
        })
        pgm.log_prob = Mock(return_value=jnp.array(-1.0))
        pgm.b_counterfactual = Mock(return_value=(
            {'x': jnp.ones((10,)), 'y': jnp.ones((10,))},
            jnp.ones((10,))
        ))
        return pgm
    
    @pytest.fixture
    def mock_dist(self):
        """Create a mock distribution object."""
        dist = Mock()
        dist.sample = Mock(return_value={
            'x': jnp.ones((10,)),
            'y': jnp.ones((10,))
        })
        dist.log_prob = Mock(return_value=jnp.array(-1.0))
        dist.b_counterfactual = Mock(return_value=(
            {'x': jnp.ones((10,)), 'y': jnp.ones((10,))},
            jnp.ones((10,))
        ))
        return dist
    
    def test_evaluate_distribution_returns_dict(self, mock_pgm, mock_dist):
        """Test that evaluate_distribution_v3 returns a dictionary."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
        
        assert isinstance(result, dict)
    
    def test_evaluate_distribution_has_expected_keys(self, mock_pgm, mock_dist):
        """Test that result has expected metric keys."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
        
        expected_keys = {
            "SWD", "NLL", "KL",
            "Counterfactual_SWD", "Counterfactual_NLL", "Counterfactual_KL"
        }
        assert set(result.keys()) == expected_keys
    
    def test_evaluate_distribution_values_are_numeric(self, mock_pgm, mock_dist):
        """Test that result values are numeric."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
        
        for value in result.values():
            assert isinstance(value, (int, float))
    
    def test_targets_frozenset_conversion(self, mock_pgm, mock_dist):
        """Test that targets are converted to frozenset if needed."""
        targets = {'x'}  # Set instead of frozenset
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                # Should not raise an error
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
        
        assert isinstance(result, dict)
    
    def test_evaluate_distribution_with_different_seeds(self, mock_pgm, mock_dist):
        """Test that different seeds produce results."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result1 = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
                result2 = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=43
                )
        
        # Both should be valid dicts
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
    
    def test_evaluate_distribution_with_custom_num_samples(self, mock_pgm, mock_dist):
        """Test with custom number of samples."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions,
                    seed=42, num_samples=500, n_condition_samples=5
                )
        
        assert isinstance(result, dict)
    
    def test_evaluate_distribution_with_causal_task(self, mock_pgm, mock_dist):
        """Test with causal task."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions,
                    seed=42, task='causal'
                )
        
        assert isinstance(result, dict)
    
    def test_evaluate_distribution_with_inference_task(self, mock_pgm, mock_dist):
        """Test with inference task."""
        targets = frozenset(['x'])
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions,
                    seed=42, task='inference'
                )
        
        assert isinstance(result, dict)


class TestEvaluationEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def mock_pgm(self):
        """Create a mock PGM object."""
        pgm = Mock()
        pgm.sample = Mock(return_value={
            'x': jnp.ones((10,)),
            'y': jnp.ones((10,))
        })
        pgm.log_prob = Mock(return_value=jnp.array(-1.0))
        pgm.b_counterfactual = Mock(return_value=(
            {'x': jnp.ones((10,)), 'y': jnp.ones((10,))},
            jnp.ones((10,))
        ))
        return pgm
    
    @pytest.fixture
    def mock_dist(self):
        """Create a mock distribution object."""
        dist = Mock()
        dist.sample = Mock(return_value={
            'x': jnp.ones((10,)),
            'y': jnp.ones((10,))
        })
        dist.log_prob = Mock(return_value=jnp.array(-1.0))
        dist.b_counterfactual = Mock(return_value=(
            {'x': jnp.ones((10,)), 'y': jnp.ones((10,))},
            jnp.ones((10,))
        ))
        return dist
    
    def test_empty_targets(self, mock_pgm, mock_dist):
        """Test with empty targets."""
        targets = frozenset()
        conditions = frozenset(['y'])
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
        
        assert isinstance(result, dict)
    
    def test_empty_conditions(self, mock_pgm, mock_dist):
        """Test with empty conditions."""
        targets = frozenset(['x'])
        conditions = frozenset()
        
        with patch('structflow.bayes.metrics.compute_swd_jax', return_value=jnp.array(0.5)):
            with patch('structflow.bayes.metrics._flatten_samples', return_value=jnp.ones((10, 1))):
                result = evaluate_distribution(
                    mock_pgm, mock_dist, targets, conditions, seed=42
                )
        
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
