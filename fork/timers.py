import time
import torch


class _CudaTimer:
    """CUDA event-based timer for GPU operations with auto-aggregation."""

    def __init__(self):
        self.events = {}  # Current active events
        self.times = {}   # Aggregated times
        self.event_list = []  # All events for finalization

    def time_section(self, name: str):
        """Context manager for timing a section."""
        return _CudaTimerContext(self, name)

    def start_section(self, name: str):
        """Start timing a section."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.events[name] = (start_event, end_event)
        self.event_list.append((name, start_event, end_event))

    def end_section(self, name: str):
        """End timing a section."""
        if name in self.events:
            start_event, end_event = self.events[name]
            end_event.record()

    def finalize(self):
        """Synchronize and compute all timings with aggregation."""
        # Single sync for all events
        if self.event_list:
            # Find the last event and sync
            last_event = self.event_list[-1][2]  # end_event of last entry
            last_event.synchronize()

        # Compute and aggregate all elapsed times
        for name, start_event, end_event in self.event_list:
            elapsed = start_event.elapsed_time(end_event)
            self.times[name] = self.times.get(name, 0.0) + elapsed

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
    """CPU-based timer for fallback with auto-aggregation."""

    def __init__(self):
        self.times = {}        # Aggregated times
        self.start_times = {}  # Current section start times

    def time_section(self, name: str):
        """Context manager for timing a section."""
        return _CPUTimerContext(self, name)

    def start_section(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.perf_counter()

    def end_section(self, name: str):
        """End timing a section with aggregation."""
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            elapsed_ms = elapsed * 1000  # Convert to ms
            # Aggregate: add to existing time if section was called before
            self.times[name] = self.times.get(name, 0.0) + elapsed_ms
            del self.start_times[name]  # Clean up

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
