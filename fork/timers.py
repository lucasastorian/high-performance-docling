import time
import torch


class _CudaTimer:
    """CUDA event-based timer for GPU operations."""

    def __init__(self):
        self.events = {}
        self.times = {}

    def time_section(self, name: str):
        """Context manager for timing a section."""
        return _CudaTimerContext(self, name)

    def start_section(self, name: str):
        """Start timing a section."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.events[name] = (start_event, end_event)

    def end_section(self, name: str):
        """End timing a section."""
        if name in self.events:
            start_event, end_event = self.events[name]
            end_event.record()

    def finalize(self):
        """Synchronize and compute all timings."""
        # Single sync for all events
        if self.events:
            last_event = None
            for start_event, end_event in self.events.values():
                last_event = end_event
            if last_event:
                last_event.synchronize()

        # Compute all elapsed times
        for name, (start_event, end_event) in self.events.items():
            self.times[name] = start_event.elapsed_time(end_event)

    def get_time(self, name: str) -> float:
        """Get timing for a section in milliseconds."""
        return self.times.get(name, 0.0)


class _CudaTimerContext:
    """Context manager for CUDA timing sections."""

    def __init__(self, timer: _CudaTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer.start_section(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.end_section(self.name)


class _CPUTimer:
    """CPU-based timer for fallback."""

    def __init__(self):
        self.times = {}
        self.start_times = {}

    def time_section(self, name: str):
        """Context manager for timing a section."""
        return _CPUTimerContext(self, name)

    def start_section(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.perf_counter()

    def end_section(self, name: str):
        """End timing a section."""
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.times[name] = elapsed * 1000  # Convert to ms

    def finalize(self):
        """No-op for CPU timer."""
        pass

    def get_time(self, name: str) -> float:
        """Get timing for a section in milliseconds."""
        return self.times.get(name, 0.0)


class _CPUTimerContext:
    """Context manager for CPU timing sections."""

    def __init__(self, timer: _CPUTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer.start_section(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.end_section(self.name)
