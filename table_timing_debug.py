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
    
    def print_summary(self, wall_time: float = None, method_name: str = ""):
        """Print timing summary with better formatting and analysis"""
        print("\n" + "="*70)
        print(f"📊 TABLE PROCESSING TIMING BREAKDOWN{' - ' + method_name if method_name else ''}")
        print("="*70)
        
        # Group timings by phase
        phase1_names = ["phase1_collect_tables", "resize_page_image", "crop_tables"]
        phase2_names = ["phase2_predict", "prepare_image_batch", "model_inference", 
                       "normalize_outputs", "match_cells", "post_process"]
        phase3_names = ["phase3_package_outputs", "finalize_predict_details", "cache_prediction"]
        
        phase1_total = 0
        phase2_total = 0
        phase3_total = 0
        other_total = 0
        
        # Print grouped timings
        print("PHASE 1 - Collection & Preprocessing:")
        for name in phase1_names:
            if name in self.timings:
                times = self.timings[name]
                total = sum(times)
                avg = total / len(times)
                phase1_total += total
                print(f"  {name:28s}: {total:7.3f}s (avg {avg*1000:6.1f}ms × {len(times)})")
        
        print(f"\nPHASE 2 - Model Inference & Matching:")
        for name in phase2_names:
            if name in self.timings:
                times = self.timings[name]
                total = sum(times)
                avg = total / len(times)
                phase2_total += total
                print(f"  {name:28s}: {total:7.3f}s (avg {avg*1000:6.1f}ms × {len(times)})")
        
        print(f"\nPHASE 3 - Output Packaging:")
        for name in phase3_names:
            if name in self.timings:
                times = self.timings[name]
                total = sum(times)
                avg = total / len(times)
                phase3_total += total
                print(f"  {name:28s}: {total:7.3f}s (avg {avg*1000:6.1f}ms × {len(times)})")
        
        print(f"\nOTHER:")
        for name, times in sorted(self.timings.items()):
            if name not in phase1_names + phase2_names + phase3_names:
                total = sum(times)
                avg = total / len(times)
                other_total += total
                print(f"  {name:28s}: {total:7.3f}s (avg {avg*1000:6.1f}ms × {len(times)})")
        
        tracked_total = phase1_total + phase2_total + phase3_total + other_total
        
        print("\n" + "-"*70)
        print(f"Phase 1 (Collection):          {phase1_total:7.3f}s ({phase1_total/tracked_total*100:5.1f}%)")
        print(f"Phase 2 (Inference & Match):   {phase2_total:7.3f}s ({phase2_total/tracked_total*100:5.1f}%)")
        print(f"Phase 3 (Output):              {phase3_total:7.3f}s ({phase3_total/tracked_total*100:5.1f}%)")
        if other_total > 0:
            print(f"Other:                         {other_total:7.3f}s ({other_total/tracked_total*100:5.1f}%)")
        print(f"{'TRACKED TOTAL':30s}: {tracked_total:7.3f}s")
        
        if wall_time:
            untracked = wall_time - tracked_total
            print(f"{'WALL TIME':30s}: {wall_time:7.3f}s")
            print(f"{'⚠️  UNTRACKED TIME':30s}: {untracked:7.3f}s ({untracked/wall_time*100:5.1f}%)")
            if untracked > 0.1:
                print("\n⚠️  Missing time likely in:")
                print("  - page.get_image() rendering")
                print("  - Token extraction & deep copying")
                print("  - Backend segmentation calls")
                print("  - Coordinate transformations")
        
        print("="*70)

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