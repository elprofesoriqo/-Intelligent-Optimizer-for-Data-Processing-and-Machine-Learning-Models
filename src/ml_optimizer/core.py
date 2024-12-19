from typing import Optional, Dict, Any, Union
import torch
import numpy as np
from dataclasses import dataclass

from .compression import TensorCompressor
from .model_selector import ModelSelector
from .trainer import ModelTrainer
from .diagnostic import ModelDiagnostics


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    compression_level: str = "medium"  # ["low", "medium", "high"]
    max_memory_usage: float = 0.8  # 80% of available memory
    enable_gpu: bool = True
    batch_size_auto_adjust: bool = True
    performance_logging: bool = True
    early_stopping: bool = True
    validation_split: float = 0.2


class MLOptimizer:
    """
    Main library class responsible for the optimization process.
    Integrates data compression, model selection, and training optimization.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the ML Optimizer with optional configuration.

        Args:
            config (OptimizationConfig, optional): Configuration settings for optimization.
                                                 If not provided, default settings will be used.
        """
        self.config = config or OptimizationConfig()
        self.diagnostic = ModelDiagnostics()
        self.compressor = TensorCompressor()
        self.model_selector = ModelSelector()
        self.trainer = ModelTrainer()

        # Check available system resources
        self.system_info = self.diagnostic.get_system_info()

    def optimize_data(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Optimizes input data based on available system resources.

        Args:
            data: Input data as numpy array or PyTorch tensor

        Returns:
            Optimized (compressed if necessary) version of the input data
        """
        self.diagnostic.log_memory_usage("Before compression")

        # Determine compression level based on data size and available memory
        data_size = data.nbytes if hasattr(data, 'nbytes') else data.element_size() * data.nelement()
        available_memory = self.system_info['available_memory']

        if data_size > available_memory * self.config.max_memory_usage:
            compression_params = self.compressor.suggest_compression_params(
                data_size=data_size,
                available_memory=available_memory
            )
            data = self.compressor.compress(data, **compression_params)

        self.diagnostic.log_memory_usage("After compression")
        return data

    def select_optimal_model(self, data: Union[np.ndarray, torch.Tensor], task_type: str) -> Dict[str, Any]:
        """
        Selects the optimal model based on data characteristics and task type.

        Args:
            data: Input data for analysis
            task_type: Type of ML task (e.g., 'classification', 'regression')

        Returns:
            Dictionary containing model configuration and initialized model
        """
        data_characteristics = self.model_selector.analyze_data(data)

        model_config = self.model_selector.select_model(
            task_type=task_type,
            data_characteristics=data_characteristics,
            system_resources=self.system_info
        )

        return model_config

    def optimize_training(self, model: torch.nn.Module,
                          data: Union[np.ndarray, torch.Tensor],
                          labels: Union[np.ndarray, torch.Tensor]) -> torch.nn.Module:
        """
        Optimizes the model training process.

        Args:
            model: PyTorch model to train
            data: Training data
            labels: Training labels/targets

        Returns:
            Trained PyTorch model
        """
        # Optimize input data
        optimized_data = self.optimize_data(data)
        optimized_labels = labels

        # Configure trainer
        trainer_config = {
            'early_stopping': self.config.early_stopping,
            'validation_split': self.config.validation_split,
            'batch_size_auto_adjust': self.config.batch_size_auto_adjust,
            'system_resources': self.system_info
        }

        # Train with resource monitoring
        trained_model = self.trainer.train(
            model=model,
            data=optimized_data,
            labels=optimized_labels,
            **trainer_config
        )

        return trained_model

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive performance report of the optimization process.

        Returns:
            Dictionary containing various performance metrics and resource usage statistics
        """
        return {
            'system_usage': self.diagnostic.get_resource_usage(),
            'compression_stats': self.compressor.get_statistics(),
            'training_metrics': self.trainer.get_training_metrics(),
            'optimization_summary': {
                'initial_memory_usage': self.diagnostic.initial_memory_usage,
                'final_memory_usage': self.diagnostic.get_current_memory_usage(),
                'training_time': self.trainer.get_total_training_time()
            }
        }


# Usage example:
"""
optimizer = MLOptimizer(OptimizationConfig(
    compression_level="high",
    enable_gpu=True
))

# Data optimization
optimized_data = optimizer.optimize_data(input_data)

# Model selection
model_config = optimizer.select_optimal_model(
    optimized_data,
    task_type="classification"
)

# Training
trained_model = optimizer.optimize_training(
    model_config['model'],
    optimized_data,
    labels
)

# Performance report
performance_report = optimizer.get_performance_report()
"""