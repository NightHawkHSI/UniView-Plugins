"""Spore plugin (plugins/spore.py): RefPack, DBPF index, binary property lists, a small fake game."""

import importlib.util
import io
import os
import struct

import pytest
from PIL import Image

from engines.sdk import Progress

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_FILE = os.path.join(ROOT, "spore.py")
GAME = r"D:\SteamLibrary\steamapps\common\Spore"

if not os.path.isfile(PLUGIN_FILE):
    pytest.skip("Spore plugin not installed (plugins/spore.py)", allow_module_level=True)


def load_module():
    spec = importlib.util.spec_from_file_location("uniview_test_spore", PLUGIN_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spore = load_module()


def test_fnv1_matches_spore_ids():
    assert spore.fnv1("") == 0x811C9DC5
    assert spore.fnv1("Animations") == spore.fnv1("animations")


def test_refpack_literals_and_overlapping_copy():
    # 3 literals "abc", then copy 6 bytes from 3 back (overlapping) -> "abcabcabc"; 0xFC ends the stream
    data = bytes([0x10, 0xFB, 0, 0, 9, 0x0F, 0x02]) + b"abc" + bytes([0xFC])
    assert spore.refpack(data) == b"abcabcabc"
    assert spore.refpack(data, limit=2)[:2] == b"ab"
    long = bytes([0x10, 0xFB, 0, 0, 8, 0xE1]) + b"12345678" + bytes([0xFC])  # 8 literals
    assert spore.refpack(long) == b"12345678"
    with pytest.raises(ValueError):
        spore.refpack(bytes([0x10, 0xFB, 0, 0, 10, 0xE1]) + b"12345678" + bytes([0xFC]))


def make_prop():
    out = struct.pack(">I", 4)
    out += struct.pack(">IHH", spore.fnv1("scale"), 0x0D, 0) + struct.pack(">f", 2.5)
    out += struct.pack(">IHH", 0x1234, 0x20, 0) + struct.pack("<IIII", 0xAA, spore.TYPE_PROP, 0xBB, 0)
    out += struct.pack(">IHH", 0x5678, 0x31, 0) + struct.pack("<3f", 1, 2, 3) + b"\0" * 4
    names = "hi".encode("utf-16-le"), "yo".encode("utf-16-le")
    out += struct.pack(">IHHII", 0x9ABC, 0x13, 0x9C, 2, 8)
    for n in names:
        out += struct.pack(">I", 2) + n
    return out


def test_prop_text():
    names = {spore.fnv1("scale"): "scale"}
    text = spore.prop_text(make_prop(), lambda v, hex_digits=8: names.get(v) or f"{v:#0{hex_digits + 2}x}")
    assert "float scale = 2.5" in text
    assert "key 0x00001234 = 0x000000bb!0x000000aa.prop" in text
    assert "vector3 0x00005678 = (1, 2, 3)" in text
    assert "string16s 0x00009abc = ['hi', 'yo']" in text


def make_package(entries):
    """A DBPF 3 package; entries = [(type, group, instance, bytes)], stored uncompressed."""
    body = b""
    rows = []
    for type_id, group, instance, data in entries:
        rows.append(struct.pack("<IIIIIIHH", type_id, group, instance, 96 + len(body), len(data) | 0x80000000,
                                len(data), 0, 1))
        body += data
    index = struct.pack("<II", 4, 0) + b"".join(rows)
    header = bytearray(96)
    header[:4] = b"DBPF"
    struct.pack_into("<II", header, 4, 3, 0)
    struct.pack_into("<I", header, 0x24, len(entries))
    struct.pack_into("<I", header, 0x2C, len(index))
    struct.pack_into("<I", header, 0x40, 96 + len(body))
    return bytes(header) + body + index


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 2), (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def test_fake_game(tmp_path):
    data = tmp_path / "Data"
    data.mkdir()
    (data / "Spore_Game.package").write_bytes(make_package([
        (spore.TYPE_PNG, spore.fnv1("ui"), spore.fnv1("logo"), png_bytes()),
        (spore.TYPE_PROP, 0, spore.fnv1("settings"), make_prop()),
        (spore.TYPE_PROP, 0, spore.fnv1("old"), make_prop()),
    ]))
    (data / "PatchData.package").write_bytes(make_package([  # loads last: its "old" replaces the first one
        (spore.TYPE_PROP, 0, spore.fnv1("old"), struct.pack(">I", 0)),
        (0x02FAC0B6, 0, spore.fnv1("readme"), b"\xef\xbb\xbfpatched text"),
    ]))
    names = tmp_path / "spore_names"
    names.mkdir()
    (names / "names.txt").write_text("logo\nui\nscale\nreadme\nsettings // a comment\nold 0x%08x\n" % spore.fnv1("old"))

    plugin = spore.PLUGIN
    assert plugin.detect(str(tmp_path)) == 95
    assert plugin.detect(str(data)) == 95  # the Data folder picked instead of the game folder
    assert plugin.detect(str(names)) == 0
    session = plugin.open(str(tmp_path), Progress())
    by_name = {a.name: a for a in session.assets}
    assert set(by_name) == {"logo.png", "settings.prop", "old.prop", "readme.locale"}
    logo = by_name["logo.png"]
    assert logo.kind == "texture" and logo.path == "Data/Spore_Game/ui/logo.png"
    assert session.image(logo).size == (4, 2)
    assert "float scale = 2.5" in session.text(by_name["settings.prop"])
    assert session.text(by_name["old.prop"]) == "\n"  # the patch's empty property list
    assert session.text(by_name["readme.locale"]) == "patched text"


@pytest.mark.skipif(not os.path.isdir(GAME), reason="Spore not installed")
def test_real_game_package_index():
    entries = spore.read_index(os.path.join(GAME, "Data", "Locale", "en-us", "Text.package"))
    assert entries and all(e.type == 0x02FAC0B6 for e in entries)
    text = spore.read_entry(entries[0]).decode("utf-8-sig", "replace")
    assert text.lstrip().startswith("#")
