"""
Performance-optimized logger wrapper that batches log messages and provides
conditional logging based on performance impact.
"""

import logging
import time
from collections import defaultdict
from functools import wraps
import threading

class PerformanceLogger:
    """A logger wrapper that batches messages and provides performance-aware logging."""
    
    def __init__(self, logger, batch_size=100, batch_timeout=1.0):
        self.logger = logger
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.batch = defaultdict(list)
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.message_count = defaultdict(int)
        
    def _should_log_immediately(self, level, msg):
        """Determine if a message should be logged immediately."""
        # Always log errors and warnings immediately
        if level >= logging.WARNING:
            return True
        # Log first occurrence of each unique message pattern
        msg_pattern = msg.split(':')[0] if ':' in msg else msg[:50]
        if self.message_count[msg_pattern] == 0:
            self.message_count[msg_pattern] += 1
            return True
        # Sample subsequent messages (log every 10th occurrence)
        self.message_count[msg_pattern] += 1
        return self.message_count[msg_pattern] % 10 == 0
    
    def _flush_batch(self):
        """Flush all batched messages."""
        with self.lock:
            for level, messages in self.batch.items():
                if messages:
                    # Combine similar messages
                    combined = self._combine_similar_messages(messages)
                    for msg in combined:
                        self.logger.log(level, msg)
            self.batch.clear()
            self.last_flush = time.time()
    
    def _combine_similar_messages(self, messages):
        """Combine similar messages to reduce log volume."""
        if len(messages) <= 5:
            return messages
        
        # Group similar messages
        groups = defaultdict(list)
        for msg in messages:
            # Extract pattern (first part before specific values)
            pattern = msg.split('[')[0] if '[' in msg else msg.split('(')[0]
            groups[pattern.strip()].append(msg)
        
        combined = []
        for pattern, group in groups.items():
            if len(group) > 3:
                combined.append(f"{pattern} (repeated {len(group)} times with different values)")
            else:
                combined.extend(group)
        
        return combined
    
    def log(self, level, msg):
        """Log a message with performance optimization."""
        if self._should_log_immediately(level, msg):
            self.logger.log(level, msg)
        else:
            with self.lock:
                self.batch[level].append(msg)
                
                # Check if we should flush
                if (len(self.batch[level]) >= self.batch_size or 
                    time.time() - self.last_flush > self.batch_timeout):
                    self._flush_batch()
    
    def debug(self, msg):
        self.log(logging.DEBUG, msg)
    
    def info(self, msg):
        self.log(logging.INFO, msg)
    
    def warning(self, msg):
        self.log(logging.WARNING, msg)
    
    def error(self, msg, exc_info=False):
        # Errors are always logged immediately
        self.logger.error(msg, exc_info=exc_info)
    
    def flush(self):
        """Force flush all pending messages."""
        self._flush_batch()

class LogContext:
    """Context manager for managing log levels and performance settings."""
    
    def __init__(self, logger_name, temp_level=None, disable=False):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.temp_level = temp_level
        self.disable = disable
        self.original_handlers = []
        
    def __enter__(self):
        if self.disable:
            self.original_level = self.logger.level
            self.logger.setLevel(logging.CRITICAL + 1)
        elif self.temp_level is not None:
            self.original_level = self.logger.level
            self.logger.setLevel(self.temp_level)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)

def timed_operation(operation_name):
    """Decorator to time operations and log only if they're slow."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Only log if operation took more than 0.1 seconds
            if elapsed > 0.1:
                logger = logging.getLogger(func.__module__)
                logger.info(f"{operation_name} took {elapsed:.2f}s")
            
            return result
        return wrapper
    return decorator