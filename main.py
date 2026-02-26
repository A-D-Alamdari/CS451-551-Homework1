#!/usr/bin/env python3
"""
Entry point for the Pacman game.

Usage:
    python main.py [options]

Run with --help for available options.
"""
import sys
from pacman.engine import read_command, run_games

if __name__ == '__main__':
    args = read_command(sys.argv[1:])
    run_games(**args)
