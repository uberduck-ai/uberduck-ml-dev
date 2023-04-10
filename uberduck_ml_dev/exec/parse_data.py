__all__ = ["FORMATS"]


import argparse
import os
from pathlib import Path

import sqlite3
from tqdm import tqdm

from ..data.cache import ensure_speaker_table, CACHE_LOCATION
from ..data.parse import (
    _cache_filelists,
    _write_db_to_csv,
    STANDARD_MULTISPEAKER,
    STANDARD_SINGLESPEAKER,
)

FORMATS = [
    STANDARD_MULTISPEAKER,
    STANDARD_SINGLESPEAKER,
]


from typing import List
import sys


def _parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path to input dataset file or directory", required=True
    )
    parser.add_argument(
        "-f", "--format", help="Input dataset format", default=STANDARD_MULTISPEAKER
    )
    parser.add_argument(
        "-n", "--name", help="Dataset name", default=STANDARD_MULTISPEAKER
    )
    parser.add_argument(
        "-d", "--database", help="Output database", default=CACHE_LOCATION
    )
    parser.add_argument("--csv_path", help="Path to save csv", default=None)
    return parser.parse_args(args)


try:
    from nbdev.imports import IN_NOTEBOOK
except:
    IN_NOTEBOOK = False

if __name__ == "__main__" and not IN_NOTEBOOK:
    args = _parse_args(sys.argv[1:])
    ensure_speaker_table(args.database)
    conn = sqlite3.connect(args.database)
    _cache_filelists(
        folder=args.input, fmt=args.format, conn=conn, dataset_name=args.name
    )
    if args.csv_path is not None:
        _write_db_to_csv(conn, args.csv_path)
