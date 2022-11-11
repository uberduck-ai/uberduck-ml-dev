__all__ = ['ensure_speaker_table']



import os
from pathlib import Path

import sqlite3

# Try catch to resolve weirdness in GitHub actions runner.
try:
    CACHE_LOCATION = Path.home() / Path(".cache/uberduck/uberduck-ml-dev.db")
except:
    pass


def _path_to_speaker_name(path: str, speaker_idx_in_path=None):
    p = Path(path)
    if speaker_idx_in_path is not None:
        return p.parts[speaker_idx_in_path]
    assert "wavs" in p.parts, f"Can't autodetect speaker name from path: {p.parts}"
    wavs_idx = p.parts.index("wavs")
    return p.parts[wavs_idx - 1]


def ensure_speaker_table(database_path):
    db_path = Path(database_path)
    if not db_path.parent.exists():
        os.makedirs(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    sql = """CREATE TABLE IF NOT EXISTS filelists (uuid TEXT,
            filelist_path TEXT,
            speaker_name TEXT,
            speaker_id INT,
            dir_path TEXT,
            rel_path TEXT,
            dataset_name TEXT)
            """
    cursor.execute(sql)
