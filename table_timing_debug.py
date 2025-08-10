"""
Debug script to understand where table processing time is spent
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TimingCollector:
    """Collects timing information for different phases"""
    timings: Dict[str, List[float]] = field(default_factory=dict)
    start_times: Dict[str, float] = field(default_factory=dict)
    
    def start(self, name: str):
        """Start timing a phase"""
        self.start_times[name] = time.time()
    
    def end(self, name: str):
        """End timing a phase"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0.0
    
    @contextmanager
    def time_block(self, name: str):
        """Context manager for timing a block"""
        self.start(name)
        try:
            yield
        finally:
            self.end(name)
    
    def print_summary(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("📊 TABLE PROCESSING TIMING BREAKDOWN")
        print("="*60)
        
        total_time = 0
        for name, times in sorted(self.timings.items()):
            if times:
                total = sum(times)
                avg = total / len(times)
                total_time += total
                print(f"{name:30s}: total={total:6.2f}s, avg={avg*1000:6.2f}ms, count={len(times)}")
        
        print("-"*60)
        print(f"{'TOTAL':30s}: {total_time:6.2f}s")
        print("="*60)

# Global timing collector
_timing_collector = TimingCollector()

def get_timing_collector():
    """Get the global timing collector"""
    return _timing_collector

def reset_timing():
    """Reset timing collector"""
    global _timing_collector
    _timing_collector = TimingCollector()

def print_timing_summary():
    """Print timing summary"""
    _timing_collector.print_summary()