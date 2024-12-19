# diagnostics.py
import numpy as np
import psutil
import torch
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ModelDiagnostics:
    def __init__(self):
        """
        Initialize diagnostics module
        """
        self.system_info = self._get_system_info()
        self.performance_metrics = {}
        self.start_time = None

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system resources information
        """
        return {
            'cpu_cores': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'ram_total': psutil.virtual_memory().total / (1024 ** 3),  # GB
            'ram_available': psutil.virtual_memory().available / (1024 ** 3),  # GB
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }

    def start_monitoring(self):
        """
        Start performance monitoring
        """
        self.start_time = time.time()
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory_usage': [] if torch.cuda.is_available() else None,
            'timestamps': []
        }

    def record_metrics(self):
        """
        Record current system metrics
        """
        if self.start_time is None:
            raise RuntimeError("Monitoring not started")

        current_time = time.time() - self.start_time

        self.performance_metrics['timestamps'].append(current_time)
        self.performance_metrics['cpu_usage'].append(psutil.cpu_percent())
        self.performance_metrics['memory_usage'].append(psutil.virtual_memory().percent)

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
            self.performance_metrics['gpu_memory_usage'].append(gpu_memory)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        report = {
            'system_info': self.system_info,
            'training_duration': time.time() - self.start_time,
            'average_metrics': {
                'cpu_usage': np.mean(self.performance_metrics['cpu_usage']),
                'memory_usage': np.mean(self.performance_metrics['memory_usage'])
            }
        }

        if torch.cuda.is_available():
            report['average_metrics']['gpu_memory_usage'] = np.mean(
                self.performance_metrics['gpu_memory_usage']
            )

        return report

    def plot_metrics(self, save_path: str = None):
        """
        Create visualizations of performance metrics
        """
        plt.figure(figsize=(15, 10))

        # CPU Usage
        plt.subplot(2, 2, 1)
        plt.plot(self.performance_metrics['timestamps'],
                 self.performance_metrics['cpu_usage'])
        plt.title('CPU Usage Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('CPU Usage (%)')

        # Memory Usage
        plt.subplot(2, 2, 2)
        plt.plot(self.performance_metrics['timestamps'],
                 self.performance_metrics['memory_usage'])
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (%)')

        if torch.cuda.is_available():
            # GPU Memory Usage
            plt.subplot(2, 2, 3)
            plt.plot(self.performance_metrics['timestamps'],
                     self.performance_metrics['gpu_memory_usage'])
            plt.title('GPU Memory Usage Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('GPU Memory (MB)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_report(self, path: str):
        """
        Save performance report to file
        """
        report = self.generate_report()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with open(f"{path}/performance_report_{timestamp}.txt", 'w') as f:
            f.write("=== Performance Report ===\n\n")

            f.write("System Information:\n")
            for key, value in report['system_info'].items():
                f.write(f"{key}: {value}\n")

            f.write("\nTraining Metrics:\n")
            f.write(f"Total Duration: {report['training_duration']:.2f} seconds\n")

            f.write("\nAverage Metrics:\n")
            for key, value in report['average_metrics'].items():
                f.write(f"{key}: {value:.2f}\n")