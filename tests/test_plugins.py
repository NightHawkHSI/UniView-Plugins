"""Every plugin in this repo loads in UniView, is listed in plugins.json, and the template (what contributors copy)
actually works."""

import importlib.util
import json
import os

import pytest
from PIL import Image

import engines
from engines.sdk import API_VERSION, EnginePlugin, GameSession, Progress

PLUGINS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_all_plugins_load():
    engines.load_plugins(PLUGINS_DIR)
    failed = [(e.name, e.origin, e.message) for e in engines.load_report() if not e.ok]
    assert not failed, failed
    ids = [p.id for p in engines.plugins()]
    assert len(ids) == len(set(ids)), f"duplicate plugin ids: {ids}"
    for plugin_id in ("source", "source2", "unreal", "fallout", "xna", "spiralknights", "wakfu", "spore"):
        assert plugin_id in ids


@pytest.mark.parametrize("plugin", engines.load_plugins(PLUGINS_DIR), ids=lambda p: p.id)
def test_plugin_contract(plugin):
    assert isinstance(plugin, EnginePlugin)
    assert plugin.id and plugin.name and plugin.version
    assert plugin.detect(os.path.join(PLUGINS_DIR, "no such folder")) in (0, None)


def test_index_matches_the_plugins():
    """plugins.json (what UniView's Plugins tab downloads) lists every plugin file with its current version."""
    spec = importlib.util.spec_from_file_location("make_index", os.path.join(PLUGINS_DIR, "_make_index.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(os.path.join(PLUGINS_DIR, "plugins.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == module.build_index(), "plugins.json is out of date: run py _make_index.py"


def load_template():
    spec = importlib.util.spec_from_file_location("uniview_template_test", os.path.join(PLUGINS_DIR, "_template.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLUGIN


def test_template_plugin_opens_a_folder(tmp_path):
    plugin = load_template()
    assert getattr(plugin, "api_version", API_VERSION) == API_VERSION
    Image.new("RGB", (8, 4), (0, 128, 255)).save(tmp_path / "wall.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "cube.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "data.bin").write_bytes(b"\0" * 16)

    assert plugin.detect(str(tmp_path)) > 0
    session = plugin.open(str(tmp_path), Progress())
    assert isinstance(session, GameSession)
    kinds = {a.path: a.kind for a in session.assets}
    assert kinds == {"wall.png": "texture", "models/cube.obj": "model", "notes.txt": "text"}
    tex = next(a for a in session.assets if a.kind == "texture")
    assert session.image(tex).size == (8, 4)
    model = next(a for a in session.assets if a.kind == "model")
    md = session.mesh(model)
    assert len(md.points) == 3 and md.triangle_count == 1
