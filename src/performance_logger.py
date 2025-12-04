"""
Performance-optimized logging handler that batches messages and provides
conditional logging based on performance impact.
"""

import logging
import time
from collections import defaultdict, deque
from functools import wraps
import threading


class PerformanceHandler(logging.Handler):
    """
    A logging handler that batches messages and provides performance-aware logging.
    Thread-safe with automatic cleanup of old batches.
    """
    
    def __init__(self, base_handler, batch_size=100, batch_timeout=1.0, max_batch_age=10.0):
        """
        Initialize the PerformanceHandler.
        
        Args:
            base_handler: The underlying handler to send batched messages to
            batch_size: Number of messages before forcing a flush
            batch_timeout: Time in seconds before forcing a flush
            max_batch_age: Maximum age of batches in seconds before cleanup
        """
        super().__init__()
        self.base_handler = base_handler
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_batch_age = max_batch_age
        
        # Thread-safe batch storage with timestamps
        self.batches = defaultdict(lambda: {'messages': deque(), 'created': time.time()})
        self.last_flush = time.time()
        self.lock = threading.RLock()
        self.message_count = defaultdict(int)
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def setFormatter(self, formatter):
        """Set formatter for both this handler and the base handler."""
        super().setFormatter(formatter)
        self.base_handler.setFormatter(formatter)
    
    def setLevel(self, level):
        """Set level for both this handler and the base handler."""
        super().setLevel(level)
        self.base_handler.setLevel(level)
    
    def _should_log_immediately(self, record):
        """Determine if a record should be logged immediately."""
        # Always log errors and warnings immediately
        if record.levelno >= logging.WARNING:
            return True
        
        # Log first occurrence of each unique message pattern
        msg_pattern = record.getMessage().split(':')[0] if ':' in record.getMessage() else record.getMessage()[:50]
        
        with self.lock:
            if self.message_count[msg_pattern] == 0:
                self.message_count[msg_pattern] += 1
                return True
            
            # Sample subsequent messages (log every 10th occurrence)
            self.message_count[msg_pattern] += 1
            return self.message_count[msg_pattern] % 10 == 0
    
    def _flush_batches(self, level_filter=None):
        """
        Flush batched messages to the base handler.
        
        Args:
            level_filter: If specified, only flush messages of this level
        """
        with self.lock:
            levels_to_flush = [level_filter] if level_filter is not None else list(self.batches.keys())
            
            for level in levels_to_flush:
                if level not in self.batches:
                    continue
                    
                batch_data = self.batches[level]
                messages = batch_data['messages']
                
                if not messages:
                    continue
                
                # Combine similar messages
                combined = self._combine_similar_messages(list(messages))
                
                # Emit combined messages through base handler
                for record in combined:
                    self.base_handler.emit(record)
                
                # Clear the batch
                messages.clear()
                batch_data['created'] = time.time()
            
            self.last_flush = time.time()
    
    def _combine_similar_messages(self, records):
        """
        Combine similar log records to reduce log volume.
        
        Args:
            records: List of LogRecord objects
            
        Returns:
            List of LogRecord objects (combined where possible)
        """
        if len(records) <= 5:
            return records
        
        # Group similar messages
        groups = defaultdict(list)
        for record in records:
            msg = record.getMessage()
            # Extract pattern (first part before specific values)
            pattern = msg.split('[')[0] if '[' in msg else msg.split('(')[0]
            groups[pattern.strip()].append(record)
        
        combined = []
        for pattern, group in groups.items():
            if len(group) > 3:
                # Create a new combined record
                representative = group[0]
                combined_record = logging.LogRecord(
                    name=representative.name,
                    level=representative.levelno,
                    pathname=representative.pathname,
                    lineno=representative.lineno,
                    msg=f"{pattern} (repeated {len(group)} times with different values)",
                    args=(),
                    exc_info=None
                )
                combined.append(combined_record)
            else:
                combined.extend(group)
        
        return combined
    
    def _periodic_cleanup(self):
        """Periodically clean up old batches and flush if needed."""
        while True:
            time.sleep(self.batch_timeout)
            
            with self.lock:
                current_time = time.time()
                
                # Flush if timeout exceeded
                if current_time - self.last_flush > self.batch_timeout:
                    self._flush_batches()
                
                # Clean up old batches
                levels_to_remove = []
                for level, batch_data in self.batches.items():
                    if current_time - batch_data['created'] > self.max_batch_age:
                        if batch_data['messages']:
                            # Flush before removing
                            self._flush_batches(level_filter=level)
                        levels_to_remove.append(level)
                
                for level in levels_to_remove:
                    del self.batches[level]
                
                # Clean up old message counts (keep only recent patterns)
                if len(self.message_count) > 1000:
                    # Reset counts that are very high (likely old patterns)
                    patterns_to_reset = [
                        pattern for pattern, count in self.message_count.items()
                        if count > 100
                    ]
                    for pattern in patterns_to_reset:
                        self.message_count[pattern] = 0
    
    def emit(self, record):
        """
        Emit a log record, either immediately or batched.
        
        Args:
            record: LogRecord to emit
        """
        try:
            # Check if we should log immediately
            if self._should_log_immediately(record):
                self.base_handler.emit(record)
            else:
                # Add to batch
                with self.lock:
                    self.batches[record.levelno]['messages'].append(record)
                    
                    # Check if we should flush this level's batch
                    if len(self.batches[record.levelno]['messages']) >= self.batch_size:
                        self._flush_batches(level_filter=record.levelno)
        except Exception:
            self.handleError(record)
    
    def flush(self):
        """Flush all batched messages."""
        self._flush_batches()
        self.base_handler.flush()
    
    def close(self):
        """Close the handler and flush remaining messages."""
        self.flush()
        self.base_handler.close()
        super().close()


class LogContext:
    """Context manager for temporarily changing log levels."""
    
    def __init__(self, logger_name, temp_level=None):
        """
        Initialize the LogContext.
        
        Args:
            logger_name: Name of the logger to modify
            temp_level: Temporary log level to set
        """
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.temp_level = temp_level
        
    def __enter__(self):
        if self.temp_level is not None:
            self.logger.setLevel(self.temp_level)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def timed_operation(operation_name, threshold=0.1):
    """
    Decorator to time operations and log only if they're slow.
    
    Args:
        operation_name: Name of the operation for logging
        threshold: Minimum time in seconds before logging (default 0.1s)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                
                # Only log if operation took more than threshold
                if elapsed > threshold:
                    logger.info(f"{operation_name} took {elapsed:.2f}s")
        
        return wrapper
    return decorator