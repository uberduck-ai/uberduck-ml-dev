__all__ = ['run', 'parse_args']



import argparse
import os
import sys

from ..utils.audio import normalize_audio, trim_audio


def run(dirname, backup, top_db):
    """Normalize all the audio files in a directory."""
    old_dirname = dirname
    if backup:
        old_dirname = f"{os.path.normpath(old_dirname)}_backup"
        os.rename(dirname, old_dirname)
    for dirpath, _, filenames in os.walk(old_dirname):
        rel_path = os.path.relpath(dirpath, old_dirname)
        for filename in filenames:
            if not filename.endswith(".wav"):
                continue
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirname, rel_path, filename)
            if not os.path.exists(os.path.join(dirname, rel_path)):
                os.makedirs(os.path.join(dirname, rel_path))
            trim_audio(old_path, new_path, top_db)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dirname",
        help="Path to the directory which contains audio files to normalize.",
    )
    parser.add_argument("--backup", dest="backup", action="store_true")
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    parser.add_argument("--top-db", type=int)
    parser.set_defaults(backup=True, top_db=20)
    return parser.parse_args(args)



try:
    from nbdev.imports import IN_NOTEBOOK
except:
    IN_NOTEBOOK = False

if __name__ == "__main__" and not IN_NOTEBOOK:
    args = parse_args(sys.argv[1:])
    run(args.dirname, args.backup, args.top_db)
