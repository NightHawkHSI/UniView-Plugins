"""Map triangle count from the BSP face table alone (fast stats for sorting/search)."""

import struct

from valve_formats import bsp


def build_bsp(faces, texinfo_flags, disp_powers=()):
    """A minimal VBSP: faces (num_edges, texinfo, dispinfo), texinfo flags, displacement powers."""
    face_data = b"".join(struct.pack("<HBBihhh", 0, 0, 0, 0, n, ti, d) + bytes(56 - 14) for n, ti, d in faces)
    tex_data = b"".join(bytes(64) + struct.pack("<ii", flags, 0) for flags in texinfo_flags)
    disp_data = b"".join(bytes(20) + struct.pack("<i", p) + bytes(176 - 24) for p in disp_powers)
    lumps = {bsp.LUMP_FACES: face_data, bsp.LUMP_TEXINFO: tex_data, bsp.LUMP_DISPINFO: disp_data}
    header_size = 8 + 64 * 16 + 4
    body, table = b"", []
    for i in range(64):
        raw = lumps.get(i, b"")
        table.append(struct.pack("<iiii", header_size + len(body), len(raw), 0, 0))
        body += raw
    return b"VBSP" + struct.pack("<i", 20) + b"".join(table) + struct.pack("<i", 1) + body


def count(data):
    return bsp.triangle_count(lambda start, length: data[start:start + length])


def test_plain_faces_and_tool_faces():
    # a quad (2 tris) and a pentagon (3 tris) on a normal surface; a nodraw quad is left out
    data = build_bsp([(4, 0, -1), (5, 0, -1), (4, 1, -1)], [0, 0x80])
    assert count(data) == (5, 9)


def test_displacements():
    # a power-2 displacement: 4 x 4 cells = 32 triangles on 5 x 5 vertices
    data = build_bsp([(4, 0, 0)], [0], disp_powers=[2])
    assert count(data) == (32, 25)


def test_not_a_map():
    import pytest
    with pytest.raises(ValueError):
        count(b"XXXX" + bytes(2000))
