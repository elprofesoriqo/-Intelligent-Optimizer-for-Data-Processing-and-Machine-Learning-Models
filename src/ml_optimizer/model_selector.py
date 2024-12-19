# model_selector.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List


class ModelSelector:
    def __init__(self):
        """
        Initialize model selector with default settings
        """
        self.supported_tasks = ['classification', 'regression', 'detection']

    def analyze_data(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze input data characteristics

        Args:
            data: Input tensor or dataset

        Returns:
            Dictionary with data characteristics
        """
        # Basic data analysis
        stats = {
            'input_shape': data.shape,
            'data_type': data.dtype,
            'num_features': data.shape[1] if len(data.shape) > 1 else 1,
            'num_samples': len(data),
            'missing_values': torch.isnan(data).sum().item(),
            'data_range': (float(data.min()), float(data.max())),
            'mean': float(data.mean()),
            'std': float(data.std())
        }

        # Complexity estimation
        stats['complexity_score'] = self._estimate_complexity(data)

        return stats

    def _estimate_complexity(self, data: torch.Tensor) -> float:
        """
        Estimate problem complexity based on data characteristics
        """
        # Simple complexity estimation based on:
        # - Number of features
        # - Data variance
        # - Data distribution

        feature_complexity = np.log(data.shape[1]) if len(data.shape) > 1 else 0
        variance_complexity = float(data.var())

        # Normalize and combine factors
        complexity_score = (feature_complexity * 0.4 +
                            variance_complexity * 0.6)

        return complexity_score

    def select_architecture(self,
                            data_stats: Dict[str, Any],
                            task_type: str,
                            num_classes: int = None) -> nn.Module:
        """
        Select optimal neural network architecture based on data characteristics
        """
        if task_type not in self.supported_tasks:
            raise ValueError(f"Task type {task_type} not supported")

        input_size = data_stats['num_features']
        complexity = data_stats['complexity_score']

        if task_type == 'classification':
            return self._create_classification_model(input_size, complexity, num_classes)
        elif task_type == 'regression':
            return self._create_regression_model(input_size, complexity)
        else:  # detection
            return self._create_detection_model(input_size, complexity)

    def _create_classification_model(self,
                                     input_size: int,
                                     complexity: float,
                                     num_classes: int) -> nn.Module:
        """
        Create classification model based on problem complexity
        """
        if complexity < 5:  # Simple problem
            return nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        elif complexity < 10:  # Medium complexity
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        else:  # High complexity
            return nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

    def _create_regression_model(self, input_size: int, complexity: float) -> nn.Module:
        """Create regression model based on problem complexity"""
        # Similar structure to classification but with single output
        layers = []

        if complexity < 5:
            layers = [64, 32, 1]
        elif complexity < 10:
            layers = [128, 64, 32, 1]
        else:
            layers = [256, 128, 64, 32, 1]

        model = nn.Sequential()
        prev_size = input_size

        for i, size in enumerate(layers):
            model.add_module(f'linear_{i}', nn.Linear(prev_size, size))
            if i < len(layers) - 1:  # No ReLU on last layer for regression
                model.add_module(f'relu_{i}', nn.ReLU())
                model.add_module(f'dropout_{i}', nn.Dropout(0.2))
            prev_size = size

        return model