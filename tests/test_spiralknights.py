"""plugins/spiralknights.py: the Clyde binary export reader, parameters, and (if installed) the real game."""
import importlib.util
import os
import struct
import zlib

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("spiralknights", os.path.join(HERE, "..", "spiralknights.py"))
sk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sk)

GAME = r"C:\Program Files (x86)\Steam\steamapps\common\Spiral Knights"


def _utf(s):
    b = s.encode()
    return struct.pack(">H", len(b)) + b


def _export(body, version=0x1002):
    return struct.pack(">IHH", 0xFACEAF0E, version, 0x1000) + zlib.compress(body)


def test_reader_objects_fields_floats_and_references():
    # root: object id 1, class 9 "a.Thing" (flags 0), 2 fields
    body = b"\x01" + b"\x09" + _utf("a.Thing") + b"\x00" + b"\x02"
    # field 1: name "pos" (string object id 2), class 10 Vector3f (final), value object id 3
    body += b"\x01" + b"\x02" + _utf("pos") + b"\x0a" + _utf("com.threerings.math.Vector3f") + b"\x01"
    body += b"\x03" + struct.pack(">3f", 1.0, 2.0, 3.0)
    # field 2: name "same" (id 4), declared Object: a reference back to the root (object id 1)
    body += b"\x02" + b"\x04" + _utf("same") + b"\x0b" + _utf("java.lang.Object") + b"\x00" + b"\x01"
    r = sk.ClydeReader(_export(body))
    root = r.read()
    assert root.cls == "a.Thing" and root["pos"] == (1.0, 2.0, 3.0) and root["same"] is root
    assert r.p == len(r.b)


def test_peek_stops_at_the_implementation():
    body = b"\x01" + b"\x09" + _utf("m.ModelConfig") + b"\x00" + b"\x01"
    body += b"\x01" + b"\x02" + _utf("implementation") + b"\x0a" + _utf("m.ModelConfig$Implementation") + b"\x00"
    body += b"\x03" + b"\x0b" + _utf("m.StaticConfig") + b"\x00" + b"\x00"
    assert sk.ClydeReader(_export(body)).peek() == ("m.ModelConfig", "m.StaticConfig")


def test_parameters_set_fields_and_reference_arguments():
    ref = sk.ClydeObject("com.threerings.config.ConfigReference")
    ref["name"] = "a.dat"
    part = sk.ClydeObject("x.Part")
    part["model"] = ref
    impl = sk.ClydeObject("x.Compound")
    impl["animationMappings"] = [part]
    root = sk.ClydeObject("x.ModelConfig")
    root["implementation"] = impl
    direct = sk.ClydeObject("com.threerings.config.Parameter$Direct")
    direct.update(name="Model", paths=["implementation.animation_mappings[0].model"])
    tint = sk.ClydeObject("com.threerings.config.Parameter$Direct")
    tint.update(name="Tint", paths=['implementation.animation_mappings[0].model["Color"]'])
    root["parameters"] = [direct, tint]
    copy = sk.apply_arguments(sk.clone(root), {"Tint": "red"})
    assert copy["implementation"]["animationMappings"][0]["model"]["arguments"] == {"Color": "red"}
    assert "arguments" not in ref  # the original is untouched
    copy = sk.apply_arguments(sk.clone(root), {"Model": "b.dat"})
    assert copy["implementation"]["animationMappings"][0]["model"] == "b.dat"


def test_transform_matrix():
    t = sk.ClydeObject("com.threerings.math.Transform3D")
    t.update(translation=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 0.7071068, 0.7071068), scale=2.0)
    m = sk._matrix(t)
    p = m @ [1.0, 0.0, 0.0, 1.0]
    assert p[:3] == pytest.approx([1.0, 4.0, 3.0], abs=1e-5)  # x turned onto y, doubled, moved


@pytest.mark.skipif(not os.path.isdir(os.path.join(GAME, "rsrc")), reason="Spiral Knights not installed")
def test_real_game_opens_and_builds_a_model():
    from engines.sdk import Progress
    plugin = sk.SpiralKnightsPlugin()
    assert plugin.detect(GAME) >= 90
    session = plugin.open(GAME, Progress())
    kinds = {a.kind for a in session.assets}
    assert {"model", "texture", "audio", "data"} <= kinds
    model = next(a for a in session.assets if a.ref == "character/npc/architect/model_architect.dat")
    mesh = session.mesh(model)
    assert len(mesh.points) > 100 and any(m.textures for m in session.materials(model))
    data = next(a for a in session.assets if a.kind == "data")
    assert session.text(data).strip()
    session.close()
