"""
Gateway Latency SLA Benchmarking and Profiling.
"""

import time
import numpy as np
from typing import Callable, Any, Dict

class LatencyProfiler:
    def __init__(self, target_sla_ms: float = 10.0):
        self.target_sla_ms = target_sla_ms

    def profile_callable(self, func: Callable, *args, iterations: int = 50, **kwargs) -> Dict[str, Any]:
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = func(*args, **kwargs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        lat_arr = np.array(latencies)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        
        return {
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "meets_sla": bool(p99 <= self.target_sla_ms),
            "target_sla_ms": self.target_sla_ms
        }
