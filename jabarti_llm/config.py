"""
config.py -- the single source of truth for every model dimension
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelConfig:
    vocab_size: int = 32_000    
    d_model: int = 768          # Embedding Dim width
    max_seq_len: int = 1024
    dropout: float = 0.1
