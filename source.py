"""Source engine plugin (Half-Life 2, TF2, Portal, CS:S, L4D2, Garry's Mod ...).

Reads the game's VPK packs and loose files: models (.mdl + .vvd + .vtx), textures (.vtf),
materials (.vmt -> $basetexture) and text files. Everything else can be exported as-is.
"""

import logging
import os
import re
import struct

import numpy as np
from PIL import Image

from engines.sdk import (
    ALBEDO, NORMAL, Asset, EnginePlugin, GameSession, Kind, Material, MeshData, TextureRef,
    cstr_at, kind_for_extension, listdir_lower, orient_to_normals, pil_image_from_bytes, view_option, z_up_to_y_up,
)
from valve_formats.vpk import VPK, VirtualFS, vpk_size

log = logging.getLogger("viewer.source")

LOOSE_DIRS = ("materials", "models", "scripts", "resource", "cfg", "particles", "maps")
LAST_MODS = ("hl2", "platform")  # shared base content: searched after the game's own folders


# --------------------------------------------------------------------------- game layout

def is_source2(path):
    return os.path.isdir(os.path.join(path, "game", "bin")) or os.path.isfile(os.path.join(path, "gameinfo.gi"))


def mod_dirs(path):
    """Folders with Source content (gameinfo.txt or *_dir.vpk), the game's own first."""
    if not os.path.isdir(path):
        return []
    here = listdir_lower(path)
    if "gameinfo.txt" in here:
        return [path]
    mods = []
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if not os.path.isdir(sub):
            continue
        files = listdir_lower(sub)
        if "gameinfo.txt" in files or any(f.endswith("_dir.vpk") for f in files):
            mods.append(sub)
    return sorted(mods, key=lambda d: (os.path.basename(d).lower() in LAST_MODS,
                                       LAST_MODS.index(os.path.basename(d).lower())
                                       if os.path.basename(d).lower() in LAST_MODS else 0))


def dir_vpks(mod):
    try:
        names = sorted(os.listdir(mod))
    except OSError:
        return []
    return [os.path.join(mod, n) for n in names if n.lower().endswith("_dir.vpk")]


# --------------------------------------------------------------------------- VTF textures

VTF_FORMATS = {
    # id: (name, bits per pixel or None for block formats, block bytes)
    0: ("RGBA8888", 32), 1: ("ABGR8888", 32), 2: ("RGB888", 24), 3: ("BGR888", 24), 4: ("RGB565", 16),
    5: ("I8", 8), 6: ("IA88", 16), 7: ("P8", 8), 8: ("A8", 8), 9: ("RGB888_BLUESCREEN", 24),
    10: ("BGR888_BLUESCREEN", 24), 11: ("ARGB8888", 32), 12: ("BGRA8888", 32), 13: ("DXT1", 4),
    14: ("DXT3", 8), 15: ("DXT5", 8), 16: ("BGRX8888", 32), 17: ("BGR565", 16), 18: ("BGRX5551", 16),
    19: ("BGRA4444", 16), 20: ("DXT1_ONEBITALPHA", 4), 21: ("BGRA5551", 16), 22: ("UV88", 16),
    23: ("UVWQ8888", 32), 24: ("RGBA16161616F", 64), 25: ("RGBA16161616", 64), 26: ("UVLX8888", 32),
    27: ("R32F", 32), 28: ("RGB323232F", 96), 29: ("RGBA32323232F", 128),
    37: ("ATI2N", 8), 38: ("ATI1N", 4),
}
BLOCK_FORMATS = {13: (1, 8), 20: (1, 8), 14: (2, 16), 15: (3, 16), 38: (4, 8), 37: (5, 16)}  # id: (bcn, block bytes)
RAW_MODES = {0: "RGBA", 1: "ABGR", 2: "RGB", 3: "BGR", 9: "RGB", 10: "BGR", 11: "ARGB", 12: "BGRA",
             16: "BGRX", 5: "L", 8: "L", 6: "LA", 23: "RGBA", 26: "RGBA"}
VTF_ENVMAP = 0x4000


def vtf_level_size(fmt, w, h, d=1):
    if fmt in BLOCK_FORMATS:
        return ((w + 3) // 4) * ((h + 3) // 4) * BLOCK_FORMATS[fmt][1] * d
    bpp = VTF_FORMATS.get(fmt, (None, 0))[1]
    if not bpp:
        raise ValueError(f"Unsupported VTF format {fmt}")
    return w * h * d * bpp // 8


def vtf_header(data):
    if data[:4] != b"VTF\0":
        raise ValueError("Not a VTF texture")
    major, minor, header_size = struct.unpack_from("<III", data, 4)
    width, height, flags, frames, first_frame = struct.unpack_from("<HHIHH", data, 16)
    fmt, = struct.unpack_from("<i", data, 52)
    mips = data[56]
    low_fmt, = struct.unpack_from("<i", data, 57)
    low_w, low_h = data[61], data[62]
    depth = struct.unpack_from("<H", data, 63)[0] if minor >= 2 else 1
    return {"minor": minor, "header_size": header_size, "w": width, "h": height, "flags": flags,
            "frames": max(frames, 1), "first_frame": first_frame, "format": fmt, "mips": max(mips, 1),
            "low_format": low_fmt, "low_w": low_w, "low_h": low_h, "depth": max(depth, 1)}


def decode_pixels(fmt, data, w, h):
    """Decode one mip level of VTF pixel data into a PIL image."""
    if fmt in BLOCK_FORMATS:
        n, _ = BLOCK_FORMATS[fmt]
        pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
        mode = {4: "L", 5: "RGB"}.get(n, "RGBA")
        img = Image.frombytes(mode, (pw, ph), data, "bcn", n)
        if n == 5:
            # BC5 normal map: red/green hold x/y, rebuild blue for a normal-map look.
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            z = np.sqrt(np.clip(1 - arr[..., 0] ** 2 - arr[..., 1] ** 2, 0, 1))
            arr[..., 2] = z
            img = Image.fromarray(((arr + 1) * 127.5).clip(0, 255).astype(np.uint8), "RGB")
        return img.crop((0, 0, w, h)) if (pw, ph) != (w, h) else img
    if fmt in RAW_MODES:
        raw = RAW_MODES[fmt]
        mode = {"L": "L", "LA": "LA", "RGB": "RGB", "BGR": "RGB", "BGRX": "RGB"}.get(raw, "RGBA")
        if raw == "BGRX":
            return Image.frombytes("RGB", (w, h), data, "raw", "BGRX")
        return Image.frombytes(mode, (w, h), data, "raw", raw)
    if fmt in (4, 17):  # RGB565 / BGR565
        v = np.frombuffer(data, "<u2", w * h).reshape(h, w).astype(np.uint32)
        high, mid, low = ((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31
        r, b = (low, high) if fmt == 4 else (high, low)
        return Image.fromarray(np.dstack([r, mid, b]).astype(np.uint8), "RGB")
    if fmt in (18, 21):  # BGRX5551 / BGRA5551
        v = np.frombuffer(data, "<u2", w * h).reshape(h, w).astype(np.uint32)
        rgba = np.dstack([((v >> 10) & 31) * 255 // 31, ((v >> 5) & 31) * 255 // 31, (v & 31) * 255 // 31,
                          np.where(v >> 15, 255, 0) if fmt == 21 else np.full_like(v, 255)]).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")
    if fmt == 19:  # BGRA4444
        v = np.frombuffer(data, "<u2", w * h).reshape(h, w).astype(np.uint32)
        rgba = np.dstack([(v >> 8) & 15, (v >> 4) & 15, v & 15, (v >> 12) & 15]).astype(np.uint8) * 17
        return Image.fromarray(rgba, "RGBA")
    if fmt == 22:  # UV88
        uv = np.frombuffer(data, np.uint8, w * h * 2).reshape(h, w, 2)
        return Image.fromarray(np.dstack([uv, np.full((h, w), 255, np.uint8)]), "RGB")
    if fmt in (24, 25, 27, 28, 29):  # HDR: simple clamp / tonemap for viewing
        dtype, ch = {24: ("<f2", 4), 25: ("<u2", 4), 27: ("<f4", 1), 28: ("<f4", 3), 29: ("<f4", 4)}[fmt]
        arr = np.frombuffer(data, dtype, w * h * ch).reshape(h, w, ch).astype(np.float32)
        if fmt == 25:
            arr = arr / 65535.0 * 16.0  # Source stores RGBA16161616 as 0..16 range HDR
        rgb = arr[..., :3] if ch >= 3 else np.repeat(arr, 3, axis=2)
        rgb = rgb / (1.0 + rgb)  # Reinhard, so bright HDR values stay visible
        out = (np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")
    raise ValueError(f"VTF format {VTF_FORMATS.get(fmt, (fmt,))[0]} isn't supported yet")


def vtf_image(data):
    hdr = vtf_header(data)
    fmt, w, h = hdr["format"], hdr["w"], hdr["h"]
    if w == 0 or h == 0:
        raise ValueError("Empty texture")
    faces = 1
    if hdr["flags"] & VTF_ENVMAP:
        faces = 6 if hdr["minor"] >= 5 or hdr["first_frame"] == 0xFFFF else 7
    # Where the high-res image data starts.
    if hdr["minor"] >= 3:
        count, = struct.unpack_from("<I", data, 68)
        offset = None
        for i in range(count):
            tag = data[80 + i * 8: 83 + i * 8]
            value, = struct.unpack_from("<I", data, 84 + i * 8)
            if tag == b"\x30\x00\x00":
                offset = value
        if offset is None:
            raise ValueError("VTF has no image data")
    else:
        offset = hdr["header_size"]
        if hdr["low_format"] >= 0 and hdr["low_w"] and hdr["low_h"]:
            offset += vtf_level_size(hdr["low_format"], hdr["low_w"], hdr["low_h"])
    # Mips are stored smallest first; mip 0 of frame 0 / face 0 / slice 0 comes last.
    per_image = hdr["frames"] * faces
    for mip in range(hdr["mips"] - 1, 0, -1):
        mw, mh, md = max(1, w >> mip), max(1, h >> mip), max(1, hdr["depth"] >> mip)
        offset += vtf_level_size(fmt, mw, mh, md) * per_image
    size = vtf_level_size(fmt, w, h)
    pixels = data[offset:offset + size]
    if len(pixels) < size:
        raise ValueError("VTF data is cut short")
    return decode_pixels(fmt, pixels, w, h)


# --------------------------------------------------------------------------- VMT materials

VMT_KEY = re.compile(r'^\s*"?(\$[a-z0-9_]+)"?\s+"?([^"\r\n]*?)"?\s*(?://.*)?$', re.I | re.M)
VMT_INCLUDE = re.compile(r'^\s*"?include"?\s+"([^"]+)"', re.I | re.M)
TEXTURE_KEYS = {"$basetexture": ALBEDO, "$bumpmap": NORMAL, "$normalmap": NORMAL, "$basetexture2": "other",
                "$detail": "other", "$envmapmask": "other", "$phongexponenttexture": "other",
                "$selfillummask": "other", "$lightwarptexture": "other", "$blendmodulatetexture": "other"}


def vmt_textures(text):
    """[(key, texture path)] from a VMT's text."""
    out = []
    for key, value in VMT_KEY.findall(text):
        key = key.lower()
        if key in TEXTURE_KEYS and value and not value.startswith("_rt_"):
            out.append((key, value.replace("\\", "/").strip("/")))
    return out


# --------------------------------------------------------------------------- MDL models

VVD_VERTEX = np.dtype([("weights", "<f4", 3), ("bones", "u1", 3), ("nbones", "u1"),
                       ("pos", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2)])


def mdl_header(data):
    if data[:4] != b"IDST":
        raise ValueError("Not a Source model (.mdl)")
    version, = struct.unpack_from("<i", data, 4)
    (tex_count, tex_offset, cd_count, cd_offset, skinref_count, skin_families, skin_offset,
     bodypart_count, bodypart_offset) = struct.unpack_from("<9i", data, 204)
    textures = []
    for i in range(tex_count):
        pos = tex_offset + i * 64
        name_off, = struct.unpack_from("<i", data, pos)
        textures.append(cstr_at(data, pos + name_off).replace("\\", "/"))
    cd_dirs = []
    for i in range(cd_count):
        off, = struct.unpack_from("<i", data, cd_offset + i * 4)
        cd_dirs.append(cstr_at(data, off).replace("\\", "/"))
    skinrefs = list(struct.unpack_from(f"<{skinref_count}h", data, skin_offset)) if skinref_count else []
    bodyparts = []
    for i in range(bodypart_count):
        bp = bodypart_offset + i * 16
        _name, n_models, _base, model_index = struct.unpack_from("<4i", data, bp)
        models = []
        for j in range(n_models):
            mp = bp + model_index + j * 148
            n_meshes, mesh_index, n_verts, vert_index = struct.unpack_from("<4i", data, mp + 72)
            meshes = []
            for k in range(n_meshes):
                material, _mi, mesh_verts, vert_offset = struct.unpack_from("<4i", data, mp + mesh_index + k * 116)
                meshes.append((material, vert_offset))
            models.append({"name": cstr_at(data, mp, 64), "vertex_start": vert_index // 48,
                           "n_verts": n_verts, "meshes": meshes})
        bodyparts.append(models)
    hull = np.frombuffer(data, "<f4", 6, 104).reshape(2, 3)
    view = np.frombuffer(data, "<f4", 6, 128).reshape(2, 3)
    box = view if np.any(view) else hull
    return {"version": version, "name": cstr_at(data, 12, 64), "textures": textures, "cd_dirs": cd_dirs,
            "skinrefs": skinrefs, "bodyparts": bodyparts, "box": box}


def _proper_rotations():
    """The 24 axis permutations with signs that are rotations (no mirroring)."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), np.float32)
            for row, (axis, sign) in enumerate(zip(perm, signs)):
                m[row, axis] = sign
            if round(float(np.linalg.det(m))) == 1:
                out.append(m)
    return out


ROTATIONS = _proper_rotations()


def upright_rotation(points, hull_min, hull_max):
    """Rotation that puts the vertices in model space.

    Skinned models keep their vertices in the pose they were modelled in (often lying down or
    turned); the game stands them up with the animation. The header's hull box is in model
    space, so pick the 90-degree rotation whose bounds match it best (identity unless clearly worse).
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = float(np.max(hull_max - hull_min)) or 1.0

    def error(m):
        corners = np.stack([m @ lo, m @ hi])
        return float(np.abs(corners.min(0) - hull_min).sum() + np.abs(corners.max(0) - hull_max).sum())

    identity = error(np.eye(3, dtype=np.float32))
    best = min(ROTATIONS, key=lambda m: (error(m), m[2, 2] != 1))  # prefer turning about Z on ties
    if identity > 0.15 * size and error(best) < 0.5 * identity:
        return best
    return None


def vvd_vertices(data):
    if data[:4] != b"IDSV":
        raise ValueError("Bad .vvd file")
    lod_verts = struct.unpack_from("<8i", data, 16)
    n_fixups, fixup_start, vert_start = struct.unpack_from("<3i", data, 48)
    verts = np.frombuffer(data, VVD_VERTEX, lod_verts[0], vert_start)
    if n_fixups:
        parts = []
        for i in range(n_fixups):
            lod, src, n = struct.unpack_from("<3i", data, fixup_start + i * 12)
            if lod >= 0:
                parts.append(verts[src:src + n])
        verts = np.concatenate(parts) if parts else verts
    return verts


def vtx_meshes(data, mdl_version):
    """For LOD 0: {(bodypart, model, mesh): [orig mesh vertex ids as triangles (flat)]}."""
    version, = struct.unpack_from("<i", data, 0)
    if version != 7:
        raise ValueError(f"Unsupported .vtx version {version}")
    n_bodyparts, bodypart_offset = struct.unpack_from("<ii", data, 28)
    sg_sizes = (33, 25) if mdl_version >= 49 else (25, 33)
    out = {}
    for b in range(n_bodyparts):
        bp = bodypart_offset + b * 8
        n_models, model_offset = struct.unpack_from("<ii", data, bp)
        for m in range(n_models):
            mp = bp + model_offset + m * 8
            n_lods, lod_offset = struct.unpack_from("<ii", data, mp)
            if not n_lods:
                continue
            lp = mp + lod_offset
            n_meshes, mesh_offset = struct.unpack_from("<ii", data, lp)
            for k in range(n_meshes):
                meshp = lp + mesh_offset + k * 9
                n_sg, sg_offset = struct.unpack_from("<ii", data, meshp)
                ids = _strip_groups(data, meshp + sg_offset, n_sg, sg_sizes)
                out[(b, m, k)] = ids
    return out


def _strip_groups(data, start, count, sizes):
    for size in sizes:
        try:
            ids = []
            for s in range(count):
                sg = start + s * size
                n_verts, vert_offset, n_idx, idx_offset = struct.unpack_from("<4i", data, sg)
                if n_verts < 0 or n_idx < 0 or n_idx % 3 or sg + idx_offset + n_idx * 2 > len(data):
                    raise ValueError("bad strip group")
                if not n_idx:
                    continue
                vtx = np.frombuffer(data, np.uint8, n_verts * 9, sg + vert_offset).reshape(n_verts, 9)
                orig = vtx[:, 4].astype(np.int64) | (vtx[:, 5].astype(np.int64) << 8)
                idx = np.frombuffer(data, "<u2", n_idx, sg + idx_offset).astype(np.int64)
                if idx.max(initial=0) >= n_verts:
                    raise ValueError("index out of range")
                ids.append(orig[idx])
            return np.concatenate(ids) if ids else np.zeros(0, np.int64)
        except (ValueError, struct.error):
            continue
    raise ValueError("Couldn't read the model's triangle data (.vtx)")


# --------------------------------------------------------------------------- session

class SourceSession(GameSession):
    def __init__(self, plugin, path, fs):
        super().__init__(plugin, path)
        self.fs = fs
        self.by_key = {}
        self._vmt_cache = {}
        self._map_cache = None
        self._scene_cache = None
        self._prop_cache = {}

    def close(self):
        self.fs.close()

    def raw(self, asset):
        return self.fs.read(asset.ref)

    def image(self, asset):
        if isinstance(asset.ref, tuple):  # texture inside a map's pakfile
            _kind, map_key, path = asset.ref
            data = self._map(map_key)[4].read(path)
            return vtf_image(data) if path.lower().endswith(".vtf") else pil_image_from_bytes(data)
        data = self.fs.read(asset.ref)
        if asset.ref.endswith(".vtf"):
            return vtf_image(data)
        return pil_image_from_bytes(data)

    # ---- models
    def _companion(self, mdl_path, exts):
        base = mdl_path[:-4]
        for ext in exts:
            if self.fs.exists(base + ext):
                return base + ext
        return None

    def _model_parts(self, asset):
        data = self.fs.read(asset.ref)
        hdr = mdl_header(data)
        vtx_path = self._companion(asset.ref, (".dx90.vtx", ".vtx", ".dx80.vtx", ".sw.vtx"))
        if vtx_path is None:
            raise ValueError("This model has no .vtx file (triangle data), so it can't be shown.")
        return hdr, vtx_meshes(self.fs.read(vtx_path), hdr["version"])

    def stats(self, asset):
        stats = {"size": asset.size}
        if asset.kind == "model" and asset.ref.endswith(".bsp"):
            # World only, counted from the face table (building the map took ~1 s each: far too slow to measure
            # hundreds of maps in the background).
            from valve_formats.bsp import triangle_count
            tris, verts = triangle_count(lambda start, length: self.fs.read(asset.ref, start, length))
            stats.update(tris=tris, verts=verts, info=f"{tris:,} tris (map)", sort=tris)
        elif asset.kind == "model":
            hdr, meshes = self._model_parts(asset)
            tris = sum(len(ids) // 3 for (b, m, _k), ids in meshes.items() if m == 0)
            verts = sum(models[0]["n_verts"] for models in hdr["bodyparts"] if models)
            stats.update(tris=tris, verts=verts, info=f"{tris:,} tris", sort=tris)
        elif asset.kind == "texture" and asset.ref.endswith(".vtf"):
            hdr = vtf_header(self.fs.read(asset.ref, 0, 80))
            w, h = hdr["w"], hdr["h"]
            stats.update(w=w, h=h, info=f"{w}×{h}", sort=w * h)
        else:
            stats.update(info="", sort=asset.size or 0)
        return stats

    def mesh(self, asset):
        if asset.ref.endswith(".bsp"):
            return self._map_mesh(asset)
        hdr, meshes = self._model_parts(asset)
        vvd_path = self._companion(asset.ref, (".vvd",))
        if vvd_path is None:
            raise ValueError("This model has no .vvd file (vertex data), so it can't be shown.")
        verts = vvd_vertices(self.fs.read(vvd_path))
        by_texture: dict[int, list] = {}  # texture index -> [index arrays]
        for b, models in enumerate(hdr["bodyparts"]):
            if not models:
                continue
            model = models[0]  # the default choice of each body part
            for k, (material, vert_offset) in enumerate(model["meshes"]):
                ids = meshes.get((b, 0, k))
                if ids is None or not len(ids):
                    continue
                tex = hdr["skinrefs"][material] if material < len(hdr["skinrefs"]) else material
                by_texture.setdefault(tex, []).append(ids + model["vertex_start"] + vert_offset)
        if not by_texture:
            raise ValueError("Model has no triangles (it may be a collision/animation-only model).")
        n = len(verts)
        pos, nrm = verts["pos"], verts["normal"]
        used = np.unique(np.concatenate([np.concatenate(p) for p in by_texture.values()]))
        used = used[used < n]
        rot = upright_rotation(pos[used], hdr["box"][0], hdr["box"][1]) if len(used) else None
        if rot is not None:
            pos, nrm = pos @ rot.T, nrm @ rot.T
        points = z_up_to_y_up(pos)
        normals = z_up_to_y_up(nrm)
        uv = verts["uv"].astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        submeshes, slots = [], []
        for tex, parts in sorted(by_texture.items()):
            tris = np.concatenate(parts).reshape(-1, 3)
            tris = tris[(tris < n).all(axis=1)]
            submeshes.append(orient_to_normals(points, tris, normals))
            slots.append(tex)
        return MeshData(points, submeshes, normals=normals, uvs={"UV0": uv}, material_slots=slots,
                        name=hdr["name"])

    def _find_vmt(self, name, cd_dirs):
        name = name.lower().replace("\\", "/")
        for folder in list(cd_dirs) + [""]:
            path = f"materials/{folder.lower().strip('/')}/{name}.vmt".replace("//", "/")
            if self.fs.exists(path):
                return path
        return None

    def _vmt(self, path, depth=0):
        """[(key, texture path)] of a VMT (following 'patch' includes)."""
        if path in self._vmt_cache:
            return self._vmt_cache[path]
        try:
            text = self.fs.read(path).decode("utf-8", "replace")
        except Exception:
            text = ""
        found = vmt_textures(text)
        include = VMT_INCLUDE.search(text)
        if include and depth < 4:
            inc = include.group(1).lower().replace("\\", "/")
            found = found + [t for t in self._vmt(inc, depth + 1) if t[0] not in {k for k, _ in found}]
        self._vmt_cache[path] = found
        return found

    def _texture_asset(self, tex_path):
        key = f"materials/{tex_path.lower()}.vtf"
        return self.by_key.get(key)

    def materials(self, asset):
        if asset.ref.endswith(".bsp"):
            return self._map_materials(asset)
        hdr = mdl_header(self.fs.read(asset.ref))
        out = []
        for name in hdr["textures"]:
            vmt = self._find_vmt(name, hdr["cd_dirs"])
            refs = []
            if vmt:
                for key, tex in self._vmt(vmt):
                    tex_asset = self._texture_asset(tex)
                    if tex_asset is not None:
                        refs.append(TextureRef(key, tex.rsplit("/", 1)[-1], tex_asset, TEXTURE_KEYS[key]))
            refs.sort(key=lambda r: r.role != ALBEDO)
            out.append(Material(name.rsplit("/", 1)[-1] + ("" if vmt else " (material not found)"), refs))
        return out

    def options_changed(self):
        self._map_cache = None
        self._scene_cache = None

    # ---- maps (.bsp)
    def _map(self, key):
        """(BSP, points, uvs, parts, pakfile, skybox leaves) of a map's world, cached (maps are big)."""
        if self._map_cache and self._map_cache[0] == key:
            return self._map_cache[1]
        from valve_formats.bsp import BSP
        bsp = BSP(self.fs.read(key))
        sky_leaves, sky_faces = set(), set()
        try:
            area = bsp.sky_area() if view_option("hide_skybox") else None
            if area is not None:
                sky_leaves = bsp.area_leaves(area)
                sky_faces = bsp.faces_in_leaves(sky_leaves)
        except Exception as e:
            log.debug("No 3D skybox found in %s: %s", key, e)
        points, uvs, parts = bsp.geometry(skip_faces=sky_faces, skip_tools=view_option("hide_tool_surfaces"))
        try:
            pak = bsp.pakfile()
        except Exception:
            pak = None
        result = (bsp, points, uvs, parts, pak, sky_leaves)
        self._map_cache = (key, result)
        self._scene_cache = None
        return result

    @staticmethod
    def _angle_matrix(pitch, yaw, roll):
        """Source's AngleMatrix (degrees) -> 3x3 rotation in Source space."""
        p, y, r = np.radians([pitch, yaw, roll])
        sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
        return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                         [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                         [-sp, sr * cp, cr * cp]], np.float32)

    def _prop(self, name):
        """(MeshData, materials) of a static prop model, cached; None if it can't be read."""
        if name not in self._prop_cache:
            asset = self.by_key.get(name.lower())
            result = None
            if asset is not None and asset.kind == "model":
                try:
                    result = (self.mesh(asset), self.materials(asset))
                except Exception as e:
                    log.debug("Static prop %s: %s", name, e)
            self._prop_cache[name] = result
        return self._prop_cache[name]

    def _map_scene(self, asset):
        """World geometry (without the 3D skybox) plus every static prop, as one MeshData + materials."""
        if self._scene_cache and self._scene_cache[0] == asset.ref:
            return self._scene_cache[1]
        bsp, points, uvs, parts, pak, sky_leaves = self._map(asset.ref)
        materials = self._world_materials(asset, bsp, parts, pak)
        all_points = [z_up_to_y_up(points)]
        all_uvs = [uvs]
        subs = [tris[:, ::-1] for _td, tris in parts]  # Source faces are clockwise
        slots = list(range(len(parts)))
        count = len(points)
        props = [p for p in bsp.static_props() if not (sky_leaves and set(p[3]) & sky_leaves)]
        conv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)  # Source (Z up) -> Y up
        material_base = {}
        placed = 0
        for name, origin, angles, _leaves in props:
            prop = self._prop(name)
            if prop is None:
                continue
            md, mats = prop
            if name not in material_base:
                material_base[name] = len(materials)
                materials.extend(mats or [Material(name.rsplit("/", 1)[-1], [])])
            rot = conv @ self._angle_matrix(*angles) @ conv.T
            all_points.append(md.points @ rot.T + conv @ origin)
            uv = next(iter(md.uvs.values()), None)
            all_uvs.append(uv if uv is not None else np.zeros((len(md.points), 2), np.float32))
            for k, tris in enumerate(md.submeshes):
                subs.append(tris + count)
                slot = md.material_slots[k] if k < len(md.material_slots) else k
                slots.append(material_base[name] + min(slot, max(len(mats) - 1, 0)))
            count += len(md.points)
            placed += 1
        log.info("Map %s: %d static props placed (%d skipped: 3D skybox or unreadable)",
                 asset.name, placed, len(bsp.static_props()) - placed)
        md = MeshData(np.concatenate(all_points), subs, uvs={"UV0": np.concatenate(all_uvs)},
                      material_slots=slots, name=asset.name.rsplit("/", 1)[-1])
        self._scene_cache = (asset.ref, (md, materials))
        return md, materials

    def _map_mesh(self, asset):
        return self._map_scene(asset)[0]

    def _map_materials(self, asset):
        return list(self._map_scene(asset)[1])

    def _read_any(self, path, pak):
        """A file from the map's embedded pakfile if it has it, else from the game."""
        if pak is not None:
            try:
                return pak.read(path)
            except KeyError:
                lowered = {n.lower(): n for n in pak.namelist()}
                if path.lower() in lowered:
                    return pak.read(lowered[path.lower()])
        return self.fs.read(path)

    def _world_materials(self, asset, bsp, parts, pak):
        names = bsp.materials()
        pak_names = {n.lower(): n for n in pak.namelist()} if pak is not None else {}
        out = []
        for td, _tris in parts:
            name = names[td][0] if td < len(names) else ""
            vmt_path = f"materials/{name.lower()}.vmt"
            refs = []
            try:
                text = self._read_any(vmt_path, pak).decode("utf-8", "replace")
            except Exception:
                text = ""
            found = vmt_textures(text)
            include = VMT_INCLUDE.search(text)
            if include:
                found += [t for t in self._vmt(include.group(1).lower().replace("\\", "/"))
                          if t[0] not in {k for k, _ in found}]
            for key, tex in found:
                tex_path = f"materials/{tex.lower()}.vtf"
                if tex_path in pak_names:
                    tex_asset = Asset("texture", f"{asset.name} :: {tex}", f"{asset.ref}::{tex_path}",
                                      ref=("pak", asset.ref, pak_names[tex_path]), path=tex_path)
                else:
                    tex_asset = self._texture_asset(tex)
                if tex_asset is not None:
                    refs.append(TextureRef(key, tex.rsplit("/", 1)[-1], tex_asset, TEXTURE_KEYS[key]))
            refs.sort(key=lambda r: r.role != ALBEDO)
            out.append(Material(name.rsplit("/", 1)[-1] or "(unnamed)", refs))
        return out

    def describe(self, asset):
        rows = [("File", self.fs.source_name(asset.ref)), ("Path", asset.path)]
        if asset.kind == "model":
            try:
                hdr = mdl_header(self.fs.read(asset.ref))
                rows.append(("MDL version", str(hdr["version"])))
                if len(hdr["bodyparts"]) > 1 or any(len(m) > 1 for m in hdr["bodyparts"]):
                    rows.append(("Body parts", f"{len(hdr['bodyparts'])} (showing the default model of each)"))
            except Exception:
                pass
        return rows


class SourcePlugin(EnginePlugin):
    id = "source"
    name = "Source"
    version = "1.0"
    author = "UniView"
    description = ("Source engine games (HL2, TF2, Portal, CS:S, L4D2, GMod...): VPK packs, "
                   ".mdl models, .vtf textures, .vmt materials.")

    def detect(self, path):
        if os.path.isfile(path):
            return 95 if path.lower().endswith("_dir.vpk") and not is_source2(os.path.dirname(path)) else 0
        if is_source2(path):
            return 0
        mods = mod_dirs(path)
        if any("gameinfo.txt" in listdir_lower(m) for m in mods):
            return 95
        return 50 if mods else 0

    def game_info(self, path):
        mods = [os.path.basename(m) for m in mod_dirs(path)]
        version = ""
        for mod in mod_dirs(path)[:1]:
            try:
                with open(os.path.join(mod, "steam.inf"), encoding="utf-8", errors="replace") as f:
                    match = re.search(r"PatchVersion\s*=\s*([\d.]+)", f.read())
                version = match.group(1) if match else ""
            except OSError:
                pass
        return {"engine_version": "", "detail": ", ".join(mods) + (f" · patch {version}" if version else "")}

    def count_files(self, path):
        if os.path.isfile(path):
            return 1
        return sum(len(dir_vpks(m)) for m in mod_dirs(path))

    def open(self, path, progress):
        fs = VirtualFS()
        if os.path.isfile(path):
            vpk_files, mods = [path], []
        else:
            mods = mod_dirs(path)
            vpk_files = [v for m in mods for v in dir_vpks(m)]
        if not vpk_files and not mods:
            raise FileNotFoundError(f"No Source game content (gameinfo.txt / *_dir.vpk) found in:\n{path}")
        failed = 0
        for mod in mods:
            progress(f"Reading loose files in {os.path.basename(mod)} ...")
            fs.add_folder(mod, LOOSE_DIRS)
        progress.files((os.path.basename(v), vpk_size(v), v) for v in vpk_files)
        for n, vpk_path in enumerate(vpk_files, 1):
            progress.file(n - 1)
            progress(f"Reading {os.path.basename(vpk_path)} ({n}/{len(vpk_files)})", n - 1, len(vpk_files))
            try:
                fs.add_vpk(VPK(vpk_path))
                log.info("Read %s (%d files)", os.path.relpath(vpk_path, os.path.dirname(path)), len(fs.packs[-1].entries))
            except Exception as e:
                failed += 1
                log.warning("Could not read %s: %s", vpk_path, e)
        progress.file(len(vpk_files))
        progress("Listing assets ...")
        session = SourceSession(self, path, fs)
        session.file_count = len(fs.packs)
        for key in fs.files:
            ext = key.rsplit(".", 1)[-1] if "." in key.rsplit("/", 1)[-1] else ""
            display = fs.display_path(key)
            kind: Kind
            if ext == "bsp":
                kind = "model"
            elif ext == "mdl":
                # Animation/sequence-only .mdl files have no mesh (.vtx) to show.
                base = key[:-4]
                has_mesh = any(base + e in fs.files for e in (".dx90.vtx", ".vtx", ".dx80.vtx", ".sw.vtx"))
                kind = "model" if has_mesh else "file"
            elif ext == "vtf":
                kind = "texture"
            else:
                kind = kind_for_extension(ext)
            if kind in ("model", "texture"):
                name = display.rsplit(".", 1)[0]
                for top in ("models/", "materials/"):
                    if name.lower().startswith(top):
                        name = name[len(top):]
            else:
                name = display
            asset = Asset(kind, name, key, uid=key, size=fs.size(key), path=display, source="",
                          ref=key, ext=ext or "bin")
            session.assets.append(asset)
            session.by_key[key] = asset
        if failed:
            session.warnings.append(f"{failed} VPK file(s) could not be read (see the console).")
        return session


PLUGIN = SourcePlugin()
