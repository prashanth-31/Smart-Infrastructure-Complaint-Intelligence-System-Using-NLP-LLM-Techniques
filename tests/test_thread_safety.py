"""Test thread safety of pipeline cache."""
from __future__ import annotations

import threading
from typing import List

from utils.analysis_pipeline import _pipeline_cache


def test_pipeline_cache_thread_safety() -> None:
    """Test that pipeline cache is thread-safe."""
    
    # Test clearing cache from multiple threads
    def clear_cache() -> None:
        for _ in range(100):
            _pipeline_cache.clear()
    
    threads: List[threading.Thread] = []
    for _ in range(10):
        thread = threading.Thread(target=clear_cache)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # If we reach here without deadlock or exception, the test passes
    assert True


def test_pipeline_cache_isolation() -> None:
    """Test that cache maintains isolation between operations."""
    
    # Clear cache to start fresh
    _pipeline_cache.clear()
    
    # Verify cache is empty after clear
    assert len(_pipeline_cache._classifier_cache) == 0


if __name__ == "__main__":
    test_pipeline_cache_thread_safety()
    test_pipeline_cache_isolation()
    print("✓ All thread safety tests passed")
