"""Plugin tests run against a UniView checkout (for engines.sdk).

UniView is found from UNIVIEW_ROOT, else the folder this repo sits in (the dev layout: UniView/plugins/),
else a sibling ../UniView folder.
"""
import os
import sys

PLUGINS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _uniview_root():
    candidates = [os.environ.get("UNIVIEW_ROOT", ""), os.path.dirname(PLUGINS_DIR),
                  os.path.join(os.path.dirname(PLUGINS_DIR), "UniView")]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "engines", "sdk.py")):
            return path
    raise RuntimeError("UniView not found: set UNIVIEW_ROOT to a UniView checkout")


for path in (_uniview_root(), PLUGINS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
