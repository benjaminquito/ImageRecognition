#!/usr/bin/env python3
"""Backward-compatible wrapper for the packaged VGG16 command."""

from image_recognition.vgg16 import main


if __name__ == "__main__":
    raise SystemExit(main())
