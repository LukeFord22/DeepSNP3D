"""
Configuration management for training
"""

# Configuration imports
try:
    from .deepspeed_config import create_deepspeed_config
except ImportError:
    # Handle missing config files gracefully
    pass

__all__ = [
    'create_deepspeed_config'
]