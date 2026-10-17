"""Writes plugins.json: the index UniView's Plugins tab reads to list, download and update plugins.

    py _make_index.py           rewrite plugins.json
    py _make_index.py --check   exit 1 if plugins.json is out of date (tests run this too)

Each plugin is a top-level .py file here (not starting with '_'). Its entry holds what the Plugins tab shows (name,
version, description, games) and every file to download: the plugin itself plus the helper packages it imports
(valve_formats/, unreal_formats/), each with its size and SHA-256 so UniView can check what it downloaded.
Needs a UniView checkout to import the plugins (UNIVIEW_ROOT, or the folder this repo sits in).
"""

import ast
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "plugins.json")
REPO = "NightHawkHSI/UniView-Plugins"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/main/"
INDEX_FORMAT = 1

# Games each plugin was written for (shown in the Plugins tab and the README). Install folder names (what
# steamapps/common/<this> is called) so UniView can match them to the games a user adds.
GAMES = {
    "source.py": ["Team Fortress 2", "Half-Life 2", "Portal", "Counter-Strike Source", "Left 4 Dead 2", "GarrysMod"],
    "source2.py": ["Counter-Strike Global Offensive", "dota 2 beta", "Deadlock", "Half-Life Alyx"],
    "unreal.py": ["Satisfactory", "Headliners", "Dead as Disco"],
    "terraria.py": ["Terraria", "Stardew Valley", "Celeste"],
    "spiralknights.py": ["Spiral Knights"],
    "wakfu.py": ["Wakfu"],
    "spore.py": ["Spore"],
    "fallout.py": ["Fallout", "Fallout 2"],
    "FalloutTac.py": ["Fallout Tactics"],
    "Ror.py": ["Risk of Rain"],
    "SeriousSam2.py": ["Serious Sam 2"],
}


def find_uniview():
    """A UniView checkout to import engines.sdk from: UNIVIEW_ROOT, the parent folder, or ../UniView."""
    for path in (os.environ.get("UNIVIEW_ROOT", ""), os.path.dirname(HERE),
                 os.path.join(os.path.dirname(HERE), "UniView")):
        if path and os.path.isfile(os.path.join(path, "engines", "sdk.py")):
            return path
    raise SystemExit("UniView not found: set UNIVIEW_ROOT to a UniView checkout")


def plugin_files():
    return sorted(f for f in os.listdir(HERE) if f.endswith(".py") and not f.startswith("_"))


def helper_packages(path):
    """Folders of this repo that a plugin imports (anywhere in the file, also inside functions)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return sorted(n for n in names if os.path.isfile(os.path.join(HERE, n, "__init__.py")))


def _file_entry(rel):
    with open(os.path.join(HERE, rel), "rb") as f:
        data = f.read()
    if b"\r\n" in data:  # GitHub serves the LF copy, so the hash of a CRLF file would never match
        raise SystemExit(f"{rel} has Windows (CRLF) line endings: save it with LF ones")
    return {"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _package_files(package):
    out = []
    for root, dirs, files in os.walk(os.path.join(HERE, package)):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.relpath(os.path.join(root, f), HERE).replace("\\", "/") for f in sorted(files)
                if f.endswith(".py")]
    return out


def _load(fname):
    spec = importlib.util.spec_from_file_location(f"uniview_index_{fname[:-3]}", os.path.join(HERE, fname))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "PLUGINS", None) or [module.PLUGIN]


def build_index():
    for path in (find_uniview(), HERE):
        if path not in sys.path:
            sys.path.insert(0, path)
    from engines.sdk import API_VERSION
    plugins = []
    for fname in plugin_files():
        first = _load(fname)[0]
        rels = [fname] + [rel for package in helper_packages(os.path.join(HERE, fname))
                          for rel in _package_files(package)]
        plugins.append({
            "file": fname, "id": first.id, "name": first.name, "version": str(first.version),
            "api_version": getattr(first, "api_version", API_VERSION), "author": getattr(first, "author", ""),
            "description": getattr(first, "description", ""), "games": GAMES.get(fname, []),
            "files": [_file_entry(rel) for rel in rels],
        })
    return {"format": INDEX_FORMAT, "repo": REPO, "base_url": BASE_URL, "plugins": plugins}


def main():
    index = build_index()
    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        with open(INDEX, encoding="utf-8") as f:
            ok = json.load(f) == index
        print("plugins.json is up to date" if ok else "plugins.json is out of date: run py _make_index.py")
        sys.exit(0 if ok else 1)
    with open(INDEX, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Wrote {INDEX}: {len(index['plugins'])} plugins")


if __name__ == "__main__":
    main()
