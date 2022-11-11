__all__ = ['parse_args']

import argparse
import sys

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config")
    args = parser.parse_args(args)
    return args
