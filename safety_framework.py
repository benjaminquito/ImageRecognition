#!/usr/bin/env python3
"""Backward-compatible wrapper for the packaged safety command."""

from image_recognition.safety import *  # noqa: F403
from image_recognition.safety import main


if __name__ == "__main__":
    raise SystemExit(main())
