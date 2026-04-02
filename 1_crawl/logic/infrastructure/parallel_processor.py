"""
Parallel Processing Module
Enables multiprocessing and concurrent operations
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Any, Dict, Optional, Iterable
import time
from .config_loader import ConfigLoader, get_logger

logger = get_logger(__name__)


class ParallelProcessor:
    """Manages parallel processing of data"""
    
    def __init__(self, 
                 num_workers: int = None,
                 use_multiprocessing: bool = None,
                 chunk_size: int = None):
        """
        Initialize parallel processor
        
        Args:
            num_workers: Number of worker processes/threads
            use_multiprocessing: Use multiprocessing (True) or threading (False)
            chunk_size: Items per chunk for each worker
        """
        if num_workers is None:
            num_workers = ConfigLoader.get("parallel.num_workers", mp.cpu_count())
        
        if use_multiprocessing is None:
            use_multiprocessing = ConfigLoader.get("parallel.use_multiprocessing", True)
        
        if chunk_size is None:
            chunk_size = ConfigLoader.get("parallel.chunk_size", 10)
        
        self.num_workers = min(num_workers, mp.cpu_count())
        self.use_multiprocessing = use_multiprocessing
        self.chunk_size = chunk_size
        
        logger.info(f"Parallel processor initialized: {self.num_workers} workers, "
                   f"multiprocessing={use_multiprocessing}, chunk_size={chunk_size}")
    
    def map(self, 
            func: Callable,
            iterable: Iterable,
            timeout: float = None) -> List[Any]:
        """
        Apply function to all items in parallel
        
        Args:
            func: Function to apply
            iterable: Iterable of items
            timeout: Timeout per item in seconds
            
        Returns:
            List of results
        """
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            start_time = time.time()
            results = list(executor.map(func, iterable, timeout=timeout, 
                                       chunksize=self.chunk_size))
            duration = time.time() - start_time
            
            logger.debug(f"Parallel map completed in {duration:.2f}s "
                        f"({len(results)} items, {duration/len(results):.3f}s per item)")
            
            return results
    
    def imap(self,
             func: Callable,
             iterable: Iterable,
             callback: Callable = None,
             timeout: float = None):
        """
        Iterator version of map (returns results as they complete)
        
        Args:
            func: Function to apply
            iterable: Iterable of items
            callback: Optional callback on each result
            timeout: Timeout per item in seconds
            
        Yields:
            Results as they complete
        """
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            futures = {executor.submit(func, item): item for item in iterable}
            
            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    if callback:
                        callback(result)
                    yield result
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    yield None
    
    def starmap(self,
                func: Callable,
                iterable: Iterable[tuple],
                timeout: float = None) -> List[Any]:
        """
        Like map but unpacks arguments from tuples
        
        Args:
            func: Function to apply
            iterable: Iterable of argument tuples
            timeout: Timeout per item in seconds
            
        Returns:
            List of results
        """
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            futures = [executor.submit(func, *args) for args in iterable]
            
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=timeout))
                except Exception as e:
                    logger.error(f"Error in parallel task: {e}")
                    results.append(None)
            
            return results


def parallelize_batch_processing(
    func: Callable,
    data_items: List[Dict[str, Any]],
    num_workers: int = None,
    chunk_size: int = None,
    progress_callback: Callable = None
) -> List[Any]:
    """
    Convenient function to parallelize batch processing
    
    Args:
        func: Function to apply to each item
        data_items: List of data items to process
        num_workers: Number of worker processes
        chunk_size: Items per worker
        progress_callback: Optional progress callback (receives count)
        
    Returns:
        List of processed results
    """
    processor = ParallelProcessor(num_workers=num_workers, chunk_size=chunk_size)
    results = []
    
    for i, result in enumerate(processor.imap(func, data_items), 1):
        results.append(result)
        if progress_callback:
            progress_callback(i)
    
    logger.info(f"Batch processing complete: {len(results)} items processed")
    return results


def process_playlist_parallel(
    playlist_tracks: List[Dict[str, Any]],
    processing_func: Callable,
    num_workers: int = None
) -> Dict[str, Any]:
    """
    Process playlist tracks in parallel
    
    Args:
        playlist_tracks: List of track dictionaries
        processing_func: Function to process each track
        num_workers: Number of workers
        
    Returns:
        Dictionary with processed tracks and stats
    """
    processor = ParallelProcessor(num_workers=num_workers)
    
    start_time = time.time()
    processed = processor.map(processing_func, playlist_tracks)
    duration = time.time() - start_time
    
    # Count successful vs failed
    successful = sum(1 for p in processed if p is not None)
    failed = len(processed) - successful
    
    result = {
        "total_tracks": len(playlist_tracks),
        "successful": successful,
        "failed": failed,
        "duration_seconds": duration,
        "avg_time_per_track": duration / len(playlist_tracks) if playlist_tracks else 0,
        "processed_tracks": processed
    }
    
    logger.info(f"Playlist processing: {successful}/{len(playlist_tracks)} successful "
               f"in {duration:.1f}s")
    
    return result


class SharedCounter:
    """Thread-safe counter for multiprocessing"""
    
    def __init__(self, value: int = 0):
        self.value = mp.Value('i', value)
        self.lock = mp.Lock()
    
    def increment(self, delta: int = 1):
        with self.lock:
            self.value.value += delta
    
    def decrement(self, delta: int = 1):
        with self.lock:
            self.value.value -= delta
    
    def get(self) -> int:
        with self.lock:
            return self.value.value
    
    def set(self, value: int):
        with self.lock:
            self.value.value = value


class ParallelBatchWriter:
    """Writes processed data to file using multiprocessing"""
    
    def __init__(self, output_file: str, queue_size: int = 100):
        self.output_file = output_file
        self.queue = Queue(maxsize=queue_size)
        self.counter = SharedCounter()
    
    def add_record(self, record: Dict[str, Any]):
        """Add record to write queue"""
        self.queue.put(record)
        self.counter.increment()
    
    def get_queue(self) -> Queue:
        """Get the queue for the writer process"""
        return self.queue
    
    def get_counter(self) -> SharedCounter:
        """Get the counter for progress tracking"""
        return self.counter
    
    @staticmethod
    def writer_process(output_file: str, queue: Queue, counter: SharedCounter):
        """Static method to run in separate process"""
        import pandas as pd
        
        batch = []
        batch_size = ConfigLoader.get("data.batch_size", 25)
        file_exists = False
        
        while True:
            try:
                record = queue.get(timeout=5)
                
                if record is None:  # Sentinel value to stop
                    if batch:
                        df = pd.DataFrame(batch)
                        df.to_csv(output_file, mode='a', header=False, 
                                 index=False, encoding='utf-8')
                    break
                
                batch.append(record)
                
                if len(batch) >= batch_size:
                    df = pd.DataFrame(batch)
                    
                    if not file_exists:
                        df.to_csv(output_file, index=False, encoding='utf-8')
                        file_exists = True
                    else:
                        df.to_csv(output_file, mode='a', header=False,
                                 index=False, encoding='utf-8')
                    
                    batch = []
                    logger.debug(f"Batch written to {output_file}")
            
            except Exception as e:
                logger.error(f"Writer process error: {e}")
                break
