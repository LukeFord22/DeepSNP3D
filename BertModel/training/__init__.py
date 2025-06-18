"""
Training package for DNA SNP detection with DeepSpeed integration
"""

# Core training components - avoid relative imports to prevent issues
try:
    # Use absolute imports or direct local imports
    pass
except ImportError:
    # Graceful handling during development/setup
    pass

__all__ = [
    'setup_training',
    'train_model',
    'evaluate'
]
