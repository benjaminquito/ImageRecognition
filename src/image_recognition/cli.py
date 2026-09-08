"""Unified command-line interface for the research package."""

from __future__ import annotations

import argparse
from typing import Sequence

from . import __version__


COMMANDS = {
    "train-mnist": "Train and evaluate the paper's MNIST CNN",
    "predict-vgg16": "Classify an image with ImageNet-pretrained VGG16",
    "safety": "Create or summarize the AV safety-study protocol",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-recognition",
        description="Reproduce the paper's image-recognition experiments.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print("\ncommands:")
        for command, description in COMMANDS.items():
            print(f"  {command:<15} {description}")
        return 0
    if args.command == "train-mnist":
        from .mnist import main as command_main
    elif args.command == "predict-vgg16":
        from .vgg16 import main as command_main
    else:
        from .safety import main as command_main
    return command_main(args.arguments)

