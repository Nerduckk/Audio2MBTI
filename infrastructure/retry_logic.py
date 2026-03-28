"""
Retry Logic and Rate Limiting Module
Implements exponential backoff, rate limiting, and resilient API calls
"""

import time
import random
from functools import wraps
from typing import Callable, Any, Optional, Tuple
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from .config_loader import ConfigLoader, get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiter with token bucket algorithm"""
    
    def __init__(self, requests_per_second: float = 2.0):
        """
        Initialize rate limiter
        
        Args:
            requests_per_second: Number of requests allowed per second
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def wait(self):
        """Wait until next request is allowed"""
        elapsed = time.time() - self.last_request_time
        wait_time = self.min_interval - elapsed
        
        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self, 
                 max_retries: int = 5,
                 initial_delay: float = 1.0,
                 max_delay: float = 300.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True,
                 strategy: str = "exponential_backoff"):
        """
        Initialize retry configuration
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_multiplier: Multiplier for exponential backoff
            jitter: Whether to add random jitter
            strategy: "exponential_backoff", "linear_backoff", or "fixed"
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.strategy = strategy
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.strategy == "exponential_backoff":
            delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        elif self.strategy == "linear_backoff":
            delay = self.initial_delay * (attempt + 1)
        else:  # fixed
            delay = self.initial_delay
        
        # Cap at max delay
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay


def retry_with_backoff(
    max_retries: int = None,
    initial_delay: float = None,
    max_delay: float = None,
    backoff_multiplier: float = None,
    jitter: bool = None,
    strategy: str = None,
    retryable_exceptions: Tuple = (RequestException, Timeout, ConnectionError)
):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_multiplier: Exponential backoff multiplier
        jitter: Whether to add random jitter
        strategy: Retry strategy (exponential_backoff, linear_backoff, fixed)
        retryable_exceptions: Tuple of exceptions to retry on
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        # Get defaults from config if not provided
        retry_config = ConfigLoader.get("retry", {})
        
        _max_retries = max_retries if max_retries is not None else retry_config.get("max_retries", 5)
        _initial_delay = initial_delay if initial_delay is not None else retry_config.get("initial_delay", 1)
        _max_delay = max_delay if max_delay is not None else retry_config.get("max_delay", 300)
        _backoff_multiplier = backoff_multiplier if backoff_multiplier is not None else retry_config.get("backoff_multiplier", 2)
        _jitter = jitter if jitter is not None else retry_config.get("jitter", True)
        _strategy = strategy if strategy is not None else retry_config.get("strategy", "exponential_backoff")
        
        config = RetryConfig(
            max_retries=_max_retries,
            initial_delay=_initial_delay,
            max_delay=_max_delay,
            backoff_multiplier=_backoff_multiplier,
            jitter=_jitter,
            strategy=_strategy
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Call to {func.__name__} failed (attempt {attempt + 1}/{config.max_retries + 1}): "
                            f"{type(e).__name__}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Call to {func.__name__} failed after {config.max_retries + 1} attempts: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


class ResilientHTTPClient:
    """HTTP client with built-in retry and rate limiting"""
    
    def __init__(self, 
                 requests_per_second: float = 2.0,
                 max_retries: int = 5,
                 timeout: int = 10):
        """
        Initialize resilient HTTP client
        
        Args:
            requests_per_second: Rate limit (requests per second)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.rate_limiter = RateLimiter(requests_per_second)
        self.retry_config = RetryConfig(max_retries=max_retries)
        self.timeout = timeout
        self.session = requests.Session()
        
        # Retry on these exceptions
        self.retryable_exceptions = (
            RequestException,
            Timeout,
            ConnectionError,
            requests.exceptions.HTTPError
        )
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with retry and rate limiting"""
        kwargs.setdefault('timeout', self.timeout)
        return self._request_with_retry('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request with retry and rate limiting"""
        kwargs.setdefault('timeout', self.timeout)
        return self._request_with_retry('POST', url, **kwargs)
    
    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Internal method to make request with retry logic"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self.rate_limiter.wait()
                
                logger.debug(f"{method} {url} (attempt {attempt + 1})")
                
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"{method} {url} failed (attempt {attempt + 1}): {type(e).__name__}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"{method} {url} failed after {attempt + 1} attempts")
        
        raise last_exception or Exception("Request failed")


def get_rate_limited_client(api_name: str = "default") -> ResilientHTTPClient:
    """
    Get a rate-limited HTTP client configured for specific API
    
    Args:
        api_name: API name (youtube, spotify, apple_music, etc.)
    
    Returns:
        Configured ResilientHTTPClient instance
    """
    rate_limit_config = ConfigLoader.get(f"rate_limiting.{api_name}", {})
    
    requests_per_second = rate_limit_config.get("requests_per_second", 2)
    max_retries = rate_limit_config.get("max_retries", 5)
    timeout = ConfigLoader.get(f"apis.{api_name}.timeout", 10)
    
    return ResilientHTTPClient(
        requests_per_second=requests_per_second,
        max_retries=max_retries,
        timeout=timeout
    )


# Predefined clients for common APIs
youtube_client = None
spotify_client = None
apple_music_client = None


def get_youtube_client() -> ResilientHTTPClient:
    """Get YouTube rate-limited client"""
    global youtube_client
    if youtube_client is None:
        youtube_client = get_rate_limited_client("youtube")
    return youtube_client


def get_spotify_client() -> ResilientHTTPClient:
    """Get Spotify rate-limited client"""
    global spotify_client
    if spotify_client is None:
        spotify_client = get_rate_limited_client("spotify")
    return spotify_client


def get_apple_music_client() -> ResilientHTTPClient:
    """Get Apple Music rate-limited client"""
    global apple_music_client
    if apple_music_client is None:
        apple_music_client = get_rate_limited_client("apple_music")
    return apple_music_client
