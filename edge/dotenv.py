"""Zero-dependency .env loader.

Reads KEY=VALUE lines from a .env file into os.environ (without overriding variables
already set in the real environment, so exported vars and CI/FC config win). It stays
dependency-free with no python-dotenv needed. Blank values are skipped so an unset
placeholder in .env never clobbers a code default.
"""

import os
from pathlib import Path


def load_dotenv(path: str = ".env", override: bool = False) -> bool:
    """Load .env if present. Returns True if a file was read, False otherwise."""
    p = Path(path)
    if not p.is_file():
        return False
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or value == "":
            continue  # don't set empty placeholders over code defaults
        if override or key not in os.environ:
            os.environ[key] = value
    return True
