#!/usr/bin/env python3
"""Backward-compatible wrapper for the packaged MNIST command."""

from image_recognition.mnist import main


if __name__ == "__main__":
    raise SystemExit(main())
