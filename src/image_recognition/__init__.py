"""Reproducible image-recognition experiments for autonomous-vehicle research."""

from .model import build_paper_cnn
from .safety import create_protocol, summarize

__all__ = ["build_paper_cnn", "create_protocol", "summarize"]
__version__ = "0.1.0"

