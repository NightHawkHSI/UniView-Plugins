"""Source engine maps (.bsp, VBSP v19-21): world geometry with UVs and materials.

Brush faces become triangles (fan triangulation), displacements (terrain) are rebuilt from
their subdivided grids, and faces with tool materials (nodraw, sky, triggers ...) are skipped.
The map's embedded pakfile (custom materials/textures) can be read with pakfile().
"""

import io
import lzma
import re
import struct
import zipfile

import numpy as np

LUMP_PLANES, LUMP_TEXDATA, LUMP_VERTEXES, LUMP_NODES, LUMP_TEXINFO, LUMP_FACES = 1, 2, 3, 5, 6, 7
LUMP_LEAFS, LUMP_LEAFFACES, LUMP_GAME = 10, 16, 35
LUMP_EDGES, LUMP_SURFEDGES, LUMP_DISPINFO, LUMP_DISP_VERTS = 12, 13, 26, 33
LUMP_PAKFILE, LUMP_TEXDATA_STRING_DATA, LUMP_TEXDATA_STRING_TABLE = 40, 43, 44
SKIP_FLAGS = 0x2 | 0x4 | 0x40 | 0x80 | 0x100 | 0x200  # sky2d, sky, trigger, nodraw, hint, skip
SKIP_MATERIALS = ("tools/", "tools\\")


def _lzma(data):
    """Valve's LZMA lump: 'LZMA', actual size, compressed size, 5 property bytes, data."""
    actual, _packed = struct.unpack_from("<II", data, 4)
    props = data[12:17]
    return lzma.decompress(props + struct.pack("<Q", actual) + data[17:], format=lzma.FORMAT_ALONE)[:actual]


def triangle_count(read):
    """(triangles, vertices) of a map's world from its face table alone - no geometry is built (for sorting and
    search). read(start, length) -> bytes of the .bsp. Tool faces (nodraw, trigger, sky...) are left out like the
    view does; the 3D skybox isn't (that needs the leaves)."""
    header = read(0, 8 + 64 * 16)
    if header[:4] != b"VBSP":
        raise ValueError("Not a Source map (.bsp)")
    lumps = [struct.unpack_from("<iiii", header, 8 + i * 16) for i in range(64)]

    def lump(index):
        off, length, _ver, _fourcc = lumps[index]
        raw = read(off, length) if length else b""
        return _lzma(raw) if raw[:4] == b"LZMA" else raw  # these three tables are small
    faces, texinfo, dispinfo = lump(LUMP_FACES), lump(LUMP_TEXINFO), lump(LUMP_DISPINFO)
    f = np.frombuffer(faces, np.dtype({"names": ["edges", "texinfo", "disp"], "formats": ["<i2", "<i2", "<i2"],
                                       "offsets": [8, 10, 12], "itemsize": 56}), len(faces) // 56)
    flags = np.frombuffer(texinfo, np.dtype({"names": ["flags"], "formats": ["<i4"], "offsets": [64],
                                             "itemsize": 72}), len(texinfo) // 72)["flags"]
    ti = f["texinfo"].astype(np.int64)
    keep = (ti >= 0) & (f["edges"] >= 3)
    keep &= ~((flags[np.clip(ti, 0, max(len(flags) - 1, 0))] & SKIP_FLAGS) != 0) if len(flags) else keep
    disp = f["disp"].astype(np.int64)
    is_disp = keep & (disp >= 0) & (f["edges"] == 4) & ((disp + 1) * 176 <= len(dispinfo))
    plain = keep & ~is_disp
    tris = int((f["edges"][plain].astype(np.int64) - 2).sum())
    verts = int(f["edges"][plain].astype(np.int64).sum())
    if is_disp.any():
        power = np.frombuffer(dispinfo, np.dtype({"names": ["power"], "formats": ["<i4"], "offsets": [20],
                                                  "itemsize": 176}), len(dispinfo) // 176)["power"]
        side = (1 << power[disp[is_disp]].astype(np.int64))
        tris += int((2 * side * side).sum())
        verts += int(((side + 1) ** 2).sum())
    return tris, verts


class BSP:
    def __init__(self, data):
        if data[:4] != b"VBSP":
            raise ValueError("Not a Source map (.bsp)")
        self.data = data
        self.version, = struct.unpack_from("<i", data, 4)
        self.lumps = [struct.unpack_from("<iiii", data, 8 + i * 16) for i in range(64)]

    def lump(self, index):
        off, length, _ver, _fourcc = self.lumps[index]
        raw = self.data[off:off + length]
        if raw[:4] == b"LZMA":
            raw = _lzma(raw)
        return raw

    def materials(self):
        """Material name of each texdata entry."""
        table = self.lump(LUMP_TEXDATA_STRING_TABLE)
        strings = self.lump(LUMP_TEXDATA_STRING_DATA)
        texdata = self.lump(LUMP_TEXDATA)
        names = []
        for i in range(len(texdata) // 32):
            string_id, width, height = struct.unpack_from("<iii", texdata, i * 32 + 12)
            off, = struct.unpack_from("<i", table, string_id * 4)
            end = strings.index(b"\0", off)
            names.append((strings[off:end].decode("utf-8", "replace").replace("\\", "/"), width, height))
        return names

    # ---- entities, 3D skybox, static props
    def entities(self):
        """[{key: value}] from the entity lump."""
        text = self.lump(0).decode("utf-8", "replace")
        out = []
        for block in re.findall(r"\{([^{}]*)\}", text):
            out.append(dict(re.findall(r'"([^"]*)"\s+"([^"]*)"', block)))
        return out

    def _leaves(self):
        raw = self.lump(LUMP_LEAFS)
        size = 56 if self.version <= 19 else 32
        return raw, size

    def leaf_at(self, point):
        """Index of the leaf containing a point (walks the BSP tree)."""
        planes = np.frombuffer(self.lump(LUMP_PLANES), np.uint8).reshape(-1, 20)
        normals = planes[:, :12].copy().view("<f4").reshape(-1, 3)
        dists = planes[:, 12:16].copy().view("<f4").ravel()
        nodes = self.lump(LUMP_NODES)
        node = 0
        for _ in range(10000):
            if node < 0:
                return -node - 1
            plane, front, back = struct.unpack_from("<iii", nodes, node * 32)
            node = front if float(np.dot(normals[plane], point)) - float(dists[plane]) >= 0 else back
        return None

    def leaf_area(self, leaf):
        raw, size = self._leaves()
        area_flags, = struct.unpack_from("<H", raw, leaf * size + 6)
        return area_flags & 0x1FF

    def sky_area(self):
        """The area of the 3D skybox (the small copy of the scenery around the sky_camera), or None."""
        for ent in self.entities():
            if ent.get("classname") == "sky_camera" and "origin" in ent:
                try:
                    origin = np.array([float(v) for v in ent["origin"].split()[:3]], np.float32)
                    leaf = self.leaf_at(origin)
                    return None if leaf is None else self.leaf_area(leaf)
                except (ValueError, struct.error):
                    return None
        return None

    def area_leaves(self, area):
        raw, size = self._leaves()
        return {i for i in range(len(raw) // size)
                if struct.unpack_from("<H", raw, i * size + 6)[0] & 0x1FF == area}

    def faces_in_leaves(self, leaves):
        raw, size = self._leaves()
        leaf_faces = np.frombuffer(self.lump(LUMP_LEAFFACES), "<u2")
        faces = set()
        for leaf in leaves:
            first, count = struct.unpack_from("<HH", raw, leaf * size + 20)
            faces.update(int(f) for f in leaf_faces[first:first + count])
        return faces

    def _game_lump(self, name):
        raw = self.lump(LUMP_GAME)
        if len(raw) < 4:
            return None, 0
        count, = struct.unpack_from("<i", raw, 0)
        want = struct.unpack("<i", name[::-1].encode())[0]
        for i in range(count):
            lump_id, _flags, version, off, length = struct.unpack_from("<iHHii", raw, 4 + i * 16)
            if lump_id == want:
                data = self.data[off:off + length]
                if data[:4] == b"LZMA":
                    data = _lzma(data)
                return data, version
        return None, 0

    def static_props(self):
        """[(model path, origin (3,), angles (pitch, yaw, roll) degrees, leaves)] from the 'sprp' game lump."""
        data, version = self._game_lump("sprp")
        if not data:
            return []
        pos = 0
        n_names, = struct.unpack_from("<i", data, pos)
        pos += 4
        names = [data[pos + i * 128:pos + (i + 1) * 128].split(b"\0")[0].decode("utf-8", "replace").replace("\\", "/")
                 for i in range(n_names)]
        pos += n_names * 128
        n_leaves, = struct.unpack_from("<i", data, pos)
        pos += 4
        leaf_list = np.frombuffer(data, "<u2", n_leaves, pos)
        pos += n_leaves * 2
        n_props, = struct.unpack_from("<i", data, pos)
        pos += 4
        if not n_props:
            return []
        size = (len(data) - pos) // n_props
        props = []
        for i in range(n_props):
            base = pos + i * size
            ox, oy, oz, pitch, yaw, roll, prop_type = struct.unpack_from("<6fH", data, base)
            leaves = ()
            if version >= 4 and size >= 30:
                first_leaf, leaf_count = struct.unpack_from("<HH", data, base + 26)
                leaves = tuple(int(x) for x in leaf_list[first_leaf:first_leaf + leaf_count])
            if prop_type < len(names):
                props.append((names[prop_type], np.array([ox, oy, oz], np.float32), (pitch, yaw, roll), leaves))
        return props

    def pakfile(self):
        raw = self.lump(LUMP_PAKFILE)
        return zipfile.ZipFile(io.BytesIO(raw)) if raw[:2] == b"PK" else None

    def geometry(self, skip_faces=(), skip_tools=True):
        """(points (N,3), uvs (N,2), [(texdata index, triangles (M,3))]) in Source units, Z up."""
        verts = np.frombuffer(self.lump(LUMP_VERTEXES), "<f4").reshape(-1, 3)
        edges = np.frombuffer(self.lump(LUMP_EDGES), "<u2").reshape(-1, 2)
        surfedges = np.frombuffer(self.lump(LUMP_SURFEDGES), "<i4")
        texinfo = np.frombuffer(self.lump(LUMP_TEXINFO), np.uint8).reshape(-1, 72)
        tex_vecs = texinfo[:, :32].copy().view("<f4").reshape(-1, 2, 4)
        tex_flags = texinfo[:, 64:68].copy().view("<i4").ravel()
        tex_data = texinfo[:, 68:72].copy().view("<i4").ravel()
        materials = self.materials()
        faces = self.lump(LUMP_FACES)
        dispinfo = self.lump(LUMP_DISPINFO)
        dispverts = np.frombuffer(self.lump(LUMP_DISP_VERTS), "<f4").reshape(-1, 5)

        points, uvs, groups = [], [], {}
        count = 0

        def uv_of(pos, ti):
            td = tex_data[ti]
            _name, w, h = materials[td] if 0 <= td < len(materials) else ("", 1, 1)
            s, t = tex_vecs[ti]
            u = (pos @ s[:3] + s[3]) / max(w, 1)
            v = (pos @ t[:3] + t[3]) / max(h, 1)
            return np.stack([u, 1.0 - v], axis=1).astype(np.float32)

        for f in range(len(faces) // 56):
            first_edge, num_edges, ti, disp = struct.unpack_from("<ihhh", faces, f * 56 + 4)
            if f in skip_faces or ti < 0 or num_edges < 3 or (skip_tools and tex_flags[ti] & SKIP_FLAGS):
                continue
            td = tex_data[ti]
            name = materials[td][0] if 0 <= td < len(materials) else ""
            if skip_tools and name.lower().startswith(SKIP_MATERIALS):
                continue
            se = surfedges[first_edge:first_edge + num_edges]
            idx = np.where(se >= 0, edges[np.abs(se), 0], edges[np.abs(se), 1])
            corners = verts[idx]
            if disp >= 0 and num_edges == 4 and (disp + 1) * 176 <= len(dispinfo):
                pos, tris = self._displacement(corners, dispinfo, disp, dispverts)
            else:
                pos = corners
                n = len(pos)
                tris = np.stack([np.zeros(n - 2, np.int64), np.arange(1, n - 1), np.arange(2, n)], axis=1)
            points.append(pos.astype(np.float32))
            uvs.append(uv_of(pos, ti))
            groups.setdefault(td, []).append(tris + count)
            count += len(pos)
        if not points:
            raise ValueError("This map has no visible world geometry.")
        parts = [(td, np.concatenate(t)) for td, t in sorted(groups.items())]
        return np.concatenate(points), np.concatenate(uvs), parts

    @staticmethod
    def _displacement(corners, dispinfo, index, dispverts):
        start = np.frombuffer(dispinfo, "<f4", 3, index * 176)
        vert_start, _tri_start, power = struct.unpack_from("<iii", dispinfo, index * 176 + 12)
        # The grid starts at the corner closest to startPosition.
        first = int(np.argmin(np.linalg.norm(corners - start, axis=1)))
        c = np.roll(corners, -first, axis=0)
        size = (1 << power) + 1
        dv = dispverts[vert_start:vert_start + size * size]
        t = np.linspace(0, 1, size, dtype=np.float32)
        # rows go from edge c0->c1 to c3->c2
        left = c[0][None, :] + (c[1] - c[0])[None, :] * t[:, None]
        right = c[3][None, :] + (c[2] - c[3])[None, :] * t[:, None]
        grid = left[:, None, :] + (right - left)[:, None, :] * t[None, :, None]
        pos = grid.reshape(-1, 3) + dv[:, :3] * dv[:, 3:4]
        r, q = np.meshgrid(np.arange(size - 1), np.arange(size - 1), indexing="ij")
        a = (r * size + q).ravel()
        b, cc, d = a + size, a + 1, a + size + 1
        tris = np.concatenate([np.stack([a, b, cc], 1), np.stack([cc, b, d], 1)])
        return pos.astype(np.float32), tris.astype(np.int64)
