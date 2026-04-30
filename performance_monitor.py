"""
Performance monitoring utilities for NPR training
VRAM tracking, throughput measurement, and bottleneck identification
"""
import torch
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict
import json
import os

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Мониторинг производительности обучения"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.epoch_times = []
        self.batch_times = []
        self.memory_usage = []
        self.throughput = []
        
        self.start_time = None
        self.epoch_start = None
        
    def start_epoch(self):
        """Начало отсчёта epoch"""
        self.epoch_start = time.time()
        self.batch_times = []
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    
    def end_epoch(self, num_samples: int) -> Dict[str, float]:
        """Завершение epoch, возврат метрик"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Throughput: samples/second
        samples_per_sec = num_samples / epoch_time
        self.throughput.append(samples_per_sec)
        
        # Memory
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'current_memory_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_memory_mb': torch.cuda.memory_reserved() / 1024**2
            }
            self.memory_usage.append(memory_stats['peak_memory_mb'])
        
        metrics = {
            'epoch_time_sec': epoch_time,
            'samples_per_sec': samples_per_sec,
            **memory_stats
        }
        
        return metrics
    
    @contextmanager
    def measure_batch(self):
        """Контекстный менеджер для измерения времени батча"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.batch_times.append(time.time() - start)
    
    def get_summary(self) -> Dict:
        """Сводка по всему обучению"""
        if not self.epoch_times:
            return {}
        
        summary = {
            'total_epochs': len(self.epoch_times),
            'avg_epoch_time_sec': sum(self.epoch_times) / len(self.epoch_times),
            'avg_throughput': sum(self.throughput) / len(self.throughput),
            'max_throughput': max(self.throughput),
            'min_throughput': min(self.throughput),
        }
        
        if self.memory_usage:
            summary.update({
                'avg_peak_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
                'max_peak_memory_mb': max(self.memory_usage),
            })
        
        return summary
    
    def save_report(self, filename: str = "performance_report.json"):
        """Сохранение отчёта"""
        report = {
            'epoch_times': self.epoch_times,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'summary': self.get_summary()
        }
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved: {filepath}")
        return filepath
    
    def log_current_stats(self, epoch: int, num_samples: int):
        """Логирование текущих статистик"""
        metrics = self.end_epoch(num_samples)
        
        logger.info(
            f"Epoch {epoch} Performance: "
            f"Time={metrics['epoch_time_sec']:.1f}s, "
            f"Throughput={metrics['samples_per_sec']:.1f} samples/s"
        )
        
        if 'peak_memory_mb' in metrics:
            logger.info(
                f"Memory: Peak={metrics['peak_memory_mb']:.0f}MB, "
                f"Current={metrics['current_memory_mb']:.0f}MB"
            )


def print_gpu_info():
    """Вывод информации о GPU"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  Compute capability: {props.major}.{props.minor}")
        logger.info(f"  Multi-processors: {props.multi_processor_count}")


def suggest_batch_size(
    model: torch.nn.Module,
    input_shape_gray: tuple = (1, 1, 256, 256),
    input_shape_rgb: tuple = (1, 3, 224, 224),
    device: str = 'cuda',
    target_memory_fraction: float = 0.85
) -> int:
    """
    Автоматический подбор batch size по доступной памяти
    """
    if not torch.cuda.is_available():
        return 32  # Default for CPU
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = total_memory * target_memory_fraction
    
    # Binary search for optimal batch size
    low, high = 1, 2048
    optimal_batch = 1
    
    while low <= high:
        mid = (low + high) // 2
        
        try:
            # Create dummy inputs
            gray = torch.randn(mid, *input_shape_gray[1:]).to(device)
            rgb = torch.randn(mid, *input_shape_rgb[1:]).to(device)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda'):
                _ = model(gray, rgb)
            
            # Check memory
            peak_memory = torch.cuda.max_memory_allocated()
            
            if peak_memory < target_memory:
                optimal_batch = mid
                low = mid + 1
            else:
                high = mid - 1
            
            # Cleanup
            del gray, rgb
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise
    
    logger.info(f"Suggested batch size: {optimal_batch} "
                f"(target memory: {target_memory/1024**3:.1f}GB)")
    return optimal_batch


@contextmanager
def cuda_memory_tracker(label: str = ""):
    """Контекстный менеджер для отслеживания памяти CUDA"""
    if not torch.cuda.is_available():
        yield
        return
    
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated() / 1024**2
    
    yield
    
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    logger.info(
        f"[{label}] Memory: "
        f"Start={start_mem:.0f}MB, "
        f"End={end_mem:.0f}MB, "
        f"Peak={peak_mem:.0f}MB, "
        f"Delta={end_mem-start_mem:+.0f}MB"
    )


if __name__ == '__main__':
    # Test
    print_gpu_info()
    
    monitor = PerformanceMonitor()
    monitor.start_epoch()
    time.sleep(0.1)
    metrics = monitor.end_epoch(num_samples=1000)
    print(f"Test metrics: {metrics}")
