"""Spore engine plugin for UniView (Maxis, 2008).

Spore keeps everything in DBPF ".package" archives (Data/*.package, Data/Locale/<lang>/*.package and the
expansions' DataEP1/ folders). A package ends with an index of entries, each keyed by three 32-bit ids:
type (the file kind), group (its folder) and instance (its name). Group and instance are FNV-1 hashes of the
lower-case names, so the real names are only known when one of the strings UniView finds hashes to them (the
game's exe, or a name list: see NAME_LISTS). Most entries are compressed with EA's RefPack (QFS).
PatchData.package overrides entries of the other packages with the same key.

What's shown:
  - .rw4 RenderWare 4 files: models (vertex/index buffers, with the texture stored inside them) and textures
  - .raster textures, .png images (DXT1/3/5 and 32-bit formats)
  - .prop property lists (binary) as readable text, .locale texts, XML / CSS / plain text files
  - .snr sounds (EA layer 3, played with vgmstream)
Everything else (animations, creature part models .gmdl, effects, .vp6 videos ...) is listed and exports as-is.
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct
import numpy as np

from engines.sdk import (
    ALBEDO, Asset, EnginePlugin, GameInfo, GameSession, Kind, Material, MeshData, Progress, TextureRef, cache_dir,
    pil_image_from_bytes,
)

TYPE_PNG, TYPE_RW4, TYPE_RASTER, TYPE_PROP = 0x2F7D0004, 0x2F4E681B, 0x2F4E681C, 0x00B1B104
# type id -> (extension, kind); None = decided from the file's first bytes
TYPES: dict[int, tuple[str, Kind | None]] = {
    TYPE_PNG: ("png", "texture"),
    TYPE_RW4: ("rw4", None),
    TYPE_RASTER: ("raster", "texture"),
    TYPE_PROP: ("prop", "text"),
    0x02FAC0B6: ("locale", "text"),
    0x01A527DB: ("snr", "audio"),
    0x376840D7: ("vp6", "file"),
    0xEE17C6AD: ("animation", "file"),
    0x00E6BCE5: ("gmdl", "file"),
    0x0248F226: ("css", "text"),
    0x024A0E52: ("spui", "file"),
    0xEA5118B0: ("effdir", "file"),
}
COMPRESSED = 0xFFFF
SNIFF_BYTES = 0x40  # decompressed bytes looked at to tell text from binary
KINDS_CACHE_VERSION = 1
RW4_MAGIC = b"\x89RW4w32\x00"
RW4_TEXTURE = 0x04000000
# Name lists: one name per line, optionally followed by its hash ("name 0x1234ABCD"); SporeModder FX's
# reg_*.txt files work. Looked for in plugins/spore_names/ and <game>/spore_names/.
NAME_LISTS = "spore_names"
NAMES_CACHE_VERSION = 1
_WORD = re.compile(rb"[A-Za-z0-9_\-~!]{3,64}")


def fnv1(name: str) -> int:
    """Spore's resource id of a name: 32-bit FNV-1 of the lower-case text."""
    h = 0x811C9DC5
    for c in name.lower().encode("latin-1", "ignore"):
        h = ((h * 0x01000193) & 0xFFFFFFFF) ^ c
    return h


# --------------------------------------------------------------------------- RefPack (QFS)

def refpack(data: bytes, limit: int = 0) -> bytes:
    """Decompress EA RefPack. limit > 0: stop once that many bytes are out (for peeking at headers)."""
    flags = data[0]
    if len(data) < 2 or data[1] != 0xFB:
        raise ValueError("Not RefPack-compressed data")
    width = 4 if flags & 0x80 else 3
    p = 2 + (width if flags & 0x01 else 0)
    size = int.from_bytes(data[p:p + width], "big")
    p += width
    out = bytearray()
    end = len(data)
    while p < end:
        b0 = data[p]
        if b0 < 0x80:
            b1 = data[p + 1]
            p += 2
            plain, count, dist = b0 & 3, ((b0 >> 2) & 7) + 3, ((b0 & 0x60) << 3) + b1 + 1
        elif b0 < 0xC0:
            b1, b2 = data[p + 1], data[p + 2]
            p += 3
            plain, count, dist = b1 >> 6, (b0 & 0x3F) + 4, ((b1 & 0x3F) << 8) + b2 + 1
        elif b0 < 0xE0:
            b1, b2, b3 = data[p + 1], data[p + 2], data[p + 3]
            p += 4
            plain, count, dist = b0 & 3, ((b0 & 0x0C) << 6) + b3 + 5, ((b0 & 0x10) << 12) + (b1 << 8) + b2 + 1
        elif b0 < 0xFC:
            p += 1
            plain, count, dist = ((b0 & 0x1F) << 2) + 4, 0, 0
        else:
            p += 1
            out += data[p:p + (b0 & 3)]
            break
        out += data[p:p + plain]
        p += plain
        if count:
            start = len(out) - dist
            if start < 0:
                raise ValueError("Corrupt RefPack data")
            if dist >= count:
                out += out[start:start + count]
            else:  # overlapping copy repeats the last `dist` bytes
                for i in range(count):
                    out.append(out[start + i])
        if limit and len(out) >= limit:
            return bytes(out)
    if len(out) != size and not limit:
        raise ValueError(f"RefPack data unpacked to {len(out)} bytes, expected {size}")
    return bytes(out)


# --------------------------------------------------------------------------- DBPF packages

class Entry:
    __slots__ = ("package", "type", "group", "instance", "offset", "size", "mem_size", "compressed")

    def __init__(self, package, type_id, group, instance, offset, size, mem_size, compressed):
        self.package, self.type, self.group, self.instance = package, type_id, group, instance
        self.offset, self.size, self.mem_size, self.compressed = offset, size, mem_size, compressed


def read_index(path: str) -> list[Entry]:
    """Entries of a DBPF 2/3 package (Spore, Darkspore)."""
    with open(path, "rb") as f:
        header = f.read(96)
        if header[:4] != b"DBPF":
            raise ValueError(f"{os.path.basename(path)} is not a DBPF package")
        count, = struct.unpack_from("<I", header, 0x24)
        index_size, = struct.unpack_from("<I", header, 0x2C)
        index_offset, = struct.unpack_from("<I", header, 0x40)
        f.seek(index_offset)
        index = f.read(index_size)
    flags, = struct.unpack_from("<I", index, 0)
    p = 4
    const: dict[int, int] = {}
    for bit in range(4):  # type, group, instance (high 32 bits), instance
        if flags & (1 << bit):
            const[bit], = struct.unpack_from("<I", index, p)
            p += 4
    fields = [bit for bit in range(4) if bit not in const]
    entries = []
    for _ in range(count):
        ids = dict(const)
        for bit in fields:
            ids[bit], = struct.unpack_from("<I", index, p)
            p += 4
        offset, size, mem_size, compressed = struct.unpack_from("<IIIH", index, p)
        p += 16
        entries.append(Entry(path, ids[0], ids[1], ids[3], offset, size & 0x7FFFFFFF, mem_size,
                             compressed == COMPRESSED))
    return entries


def read_entry(entry: Entry, limit: int = 0) -> bytes:
    """An entry's bytes; limit > 0: only (at least) the first `limit` bytes, which saves unpacking the rest."""
    with open(entry.package, "rb") as f:
        f.seek(entry.offset)
        data = f.read(min(entry.size, limit) if limit and not entry.compressed else entry.size)
    return refpack(data, limit) if entry.compressed else data


# --------------------------------------------------------------------------- textures

def decode_texture(fmt: int, w: int, h: int, data: bytes):
    from PIL import Image
    fourcc = struct.pack("<I", fmt)
    bcn = {b"DXT1": 1, b"DXT3": 2, b"DXT5": 3}.get(fourcc)
    if bcn:
        return Image.frombytes("RGBA", (w, h), data, "bcn", bcn)
    if fmt in (0x15, 0x16):  # D3DFMT_A8R8G8B8 / X8R8G8B8: BGRA in memory
        img = Image.frombytes("RGBA", (w, h), data[:w * h * 4], "raw", "BGRA")
        return img.convert("RGB") if fmt == 0x16 else img
    if fmt == 0x32:  # L8
        return Image.frombytes("L", (w, h), data[:w * h])
    if fmt == 0x1C:  # A8
        return Image.frombytes("L", (w, h), data[:w * h])
    raise NotImplementedError(f"Texture format {fourcc!r} / {fmt:#x} isn't supported yet")


def raster_image(data: bytes):
    """A .raster file: version, width, height, mip count, bytes per pixel, format, then [size, pixels] per mip."""
    _version, w, h, _mips, _bpp, fmt, size = struct.unpack_from("<7I", data, 0)
    return decode_texture(fmt, w, h, data[28:28 + size])


class RW4:
    """A RenderWare 4 file: a header, objects ("sections", each with a type code) and raw data blocks."""

    def __init__(self, data: bytes):
        if data[:8] != RW4_MAGIC:
            raise ValueError("Not a RenderWare 4 file")
        self.data = data
        self.kind, count, _count2, _align, _zero, table, _manifest, _a, _b, _c, self.data_start = \
            struct.unpack_from("<11I", data, 0x1C)
        # (offset, unknown, size, alignment, type index, type code); data blocks (0x10030) count from data_start
        self.sections = [struct.unpack_from("<6I", data, table + 24 * i) for i in range(count)]

    def section(self, index: int) -> bytes:
        offset, _u, size, _al, _ti, code = self.sections[index]
        if code == 0x10030:
            offset += self.data_start
        return self.data[offset:offset + size]

    def of_type(self, code: int) -> list[int]:
        return [i for i, s in enumerate(self.sections) if s[5] == code]

    def raster(self, index: int):
        raw = self.section(index)
        fmt, = struct.unpack_from("<I", raw, 0)
        w, h = struct.unpack_from("<HH", raw, 0x0C)
        block, = struct.unpack_from("<I", raw, 0x1C)
        return decode_texture(fmt, w, h, self.section(block))

    def rasters(self) -> list[int]:
        """Raster sections, biggest first (a model's own texture before any small placeholders)."""
        def area(i):
            w, h = struct.unpack_from("<HH", self.section(i), 0x0C)
            return w * h
        return sorted(self.of_type(0x20003), key=area, reverse=True)

    def mesh(self) -> MeshData:
        points, uvs, submeshes = [], [], []
        base = 0
        vb_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        for i in self.of_type(0x20009):
            raw = self.section(i)
            _u, prim, ib, _tris, vb_count, first_index, index_count, _first_vertex, _vertex_count = \
                struct.unpack_from("<9I", raw, 0)
            refs = struct.unpack_from(f"<{vb_count}I", raw, 36)
            if prim != 4:
                continue  # only triangle lists
            vb = next((r for r in refs if r < len(self.sections) and self.sections[r][5] == 0x20005), None)
            if vb is None:
                raise NotImplementedError("This model keeps its vertices with blend shapes, which UniView "
                                          "can't read yet (export saves the .rw4 as-is)")
            if vb not in vb_cache:
                vb_cache[vb] = self._vertices(vb)
            pos, uv = vb_cache[vb]
            indices = self._indices(ib)[first_index:first_index + index_count]
            points.append(pos)
            uvs.append(uv if uv is not None else np.zeros((len(pos), 2), np.float32))
            submeshes.append(indices.reshape(-1, 3).astype(np.int64) + base)
            base += len(pos)
        if not points:
            raise NotImplementedError("No triangle meshes in this file")
        tris = [s[:, ::-1] for s in submeshes]  # Direct3D winding (clockwise) -> counter-clockwise
        pts = np.concatenate(points)
        pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)  # Z up -> Y up
        uv = np.concatenate(uvs)
        uv[:, 1] = 1.0 - uv[:, 1]
        return MeshData(pts, tris, uvs={"UV0": uv})

    def _indices(self, ib: int) -> np.ndarray:
        _ptr, start, count, _u, fmt, _prim, block = struct.unpack_from("<7I", self.section(ib), 0)
        dtype = "<u2" if fmt == 0x65 else "<u4"
        return np.frombuffer(self.section(block), dtype)[start:start + count]

    def _vertices(self, vb: int) -> tuple[np.ndarray, np.ndarray | None]:
        desc, _ptr, base, count, _u, stride, block = struct.unpack_from("<7I", self.section(vb), 0)
        layout = self.section(desc)
        n_elements, = struct.unpack_from("<H", layout, 0x0C)
        buf = np.frombuffer(self.section(block), np.uint8)
        rows = buf[base * stride:(base + count) * stride].reshape(count, stride)
        pos = uv = None
        for k in range(n_elements):
            _stream, offset, dtype, _method, usage, usage_index = struct.unpack_from("<HHBBBB", layout, 0x18 + 12 * k)
            if usage == 0 and dtype == 2:  # POSITION, FLOAT3
                pos = rows[:, offset:offset + 12].copy().view("<f4").reshape(count, 3)
            elif usage == 5 and usage_index == 0:  # TEXCOORD0
                if dtype == 1:  # FLOAT2
                    uv = rows[:, offset:offset + 8].copy().view("<f4").reshape(count, 2).copy()
                elif dtype == 15:  # FLOAT16_2
                    uv = rows[:, offset:offset + 4].copy().view("<f2").reshape(count, 2).astype(np.float32)
        if pos is None:
            raise NotImplementedError("This model's vertices have no float positions")
        return pos.astype(np.float32), uv


# --------------------------------------------------------------------------- property lists

PROP_TYPES = {0x01: "bool", 0x02: "int8", 0x03: "uint8", 0x05: "int16", 0x06: "uint16", 0x09: "int32",
              0x0A: "uint32", 0x0B: "int64", 0x0C: "uint64", 0x0D: "float", 0x12: "string8", 0x13: "string16",
              0x20: "key", 0x22: "text", 0x30: "vector2", 0x31: "vector3", 0x32: "colorRGB", 0x33: "vector4",
              0x34: "colorRGBA", 0x38: "transform", 0x39: "bbox"}
# Bytes of one value stored on its own (vectors and keys are padded to 16) / in an array uses the stored size.
PROP_SINGLE = {0x01: 1, 0x02: 1, 0x03: 1, 0x05: 2, 0x06: 2, 0x09: 4, 0x0A: 4, 0x0B: 8, 0x0C: 8, 0x0D: 4,
               0x20: 16, 0x22: 520, 0x30: 16, 0x31: 16, 0x32: 16, 0x33: 16, 0x34: 16, 0x38: 56, 0x39: 24}
PROP_ARRAY = 0x30


def _prop_value(typ: int, raw: bytes, name) -> str:
    # Scalars are big-endian; vectors, colors, keys and boxes are little-endian.
    if typ == 0x01:
        return "true" if raw[0] else "false"
    scalar = {0x02: ">b", 0x03: ">B", 0x05: ">h", 0x06: ">H", 0x09: ">i", 0x0A: ">I", 0x0B: ">q", 0x0C: ">Q",
              0x0D: ">f"}.get(typ)
    if scalar:
        value = struct.unpack_from(scalar, raw)[0]
        if typ == 0x0A and value > 0xFFFF:
            return name(value, hex_digits=8)
        return f"{value:g}" if typ == 0x0D else str(value)
    if typ == 0x20:
        instance, type_id, group = struct.unpack_from("<III", raw)
        ext = TYPES.get(type_id, (f"{type_id:#010x}",))[0]
        return f"{name(group)}!{name(instance)}.{ext}"
    if typ == 0x22:
        table, instance = struct.unpack_from("<II", raw)
        text = raw[8:].decode("utf-16-le", "replace").split("\0")[0]
        return f"({name(table)}!{instance:#010x}) {text!r}" if text else f"({name(table)}!{instance:#010x})"
    if typ == 0x38:  # flags, offset, scale, 3x3 rotation
        offset = struct.unpack_from("<3f", raw, 4)
        scale, = struct.unpack_from("<f", raw, 16)
        rot = struct.unpack_from("<9f", raw, 20)
        return (f"(offset ({', '.join(f'{v:g}' for v in offset)}), scale {scale:g}, "
                f"rotation ({', '.join(f'{v:g}' for v in rot)}))")
    floats = {0x30: 2, 0x31: 3, 0x32: 3, 0x33: 4, 0x34: 4, 0x39: 6}.get(typ)
    if floats:
        values = struct.unpack_from(f"<{floats}f", raw)
        return "(" + ", ".join(f"{v:g}" for v in values) + ")"
    return raw.hex()


def prop_text(data: bytes, name) -> str:
    """A binary .prop as 'type name = value' lines (names where known, else the hash)."""
    count, = struct.unpack_from(">I", data, 0)
    p = 4
    lines = []
    for _ in range(count):
        pid, typ, flags = struct.unpack_from(">IHH", data, p)
        p += 8
        tname = PROP_TYPES.get(typ, f"type{typ:#x}")
        if typ in (0x12, 0x13):
            width, enc = (2, "utf-16-le") if typ == 0x13 else (1, "latin-1")
            n = 1
            if flags & PROP_ARRAY:
                n, _size = struct.unpack_from(">II", data, p)
                p += 8
            values = []
            for _ in range(n):
                length, = struct.unpack_from(">I", data, p)
                p += 4
                values.append(repr(data[p:p + length * width].decode(enc, "replace")))
                p += length * width
        elif flags & PROP_ARRAY:
            n, size = struct.unpack_from(">II", data, p)
            p += 8
            values = [_prop_value(typ, data[p + k * size:p + (k + 1) * size], name) for k in range(n)]
            p += n * size
        else:
            if typ not in PROP_SINGLE:
                raise ValueError(f"Unknown property type {typ:#x}")
            size = PROP_SINGLE[typ]
            values = [_prop_value(typ, data[p:p + size], name)]
            p += size
        if flags & PROP_ARRAY:
            lines.append(f"{tname}s {name(pid)} = [{', '.join(values)}]" if values else f"{tname}s {name(pid)} = []")
        else:
            lines.append(f"{tname} {name(pid)} = {values[0]}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- session

def rw4_kind(entry: Entry) -> Kind:
    """texture, model (has a mesh) or file (materials only, skeletons, collision shapes) for a .rw4 entry.

    Only unpacks the file up to its section table (the big vertex/pixel data comes after it)."""
    head = read_entry(entry, 0x40)
    if head[:8] != RW4_MAGIC or len(head) < 0x34:
        return "file"
    kind, count, _count2, _align, _zero, table = struct.unpack_from("<6I", head, 0x1C)
    if kind == RW4_TEXTURE:
        return "texture"
    data = read_entry(entry, table + 24 * count)
    codes = {struct.unpack_from("<I", data, table + 24 * i + 20)[0] for i in range(count)}
    return "model" if 0x20009 in codes else "file"


def _looks_like_text(head: bytes) -> str:
    """'xml' / 'txt' if a file's first bytes are text, else ''."""
    head = head.removeprefix(b"\xef\xbb\xbf")
    if not head:
        return ""
    if head.lstrip().startswith((b"<?xml", b"<")):
        return "xml"
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in head)
    return "txt" if printable >= len(head) * 0.95 else ""


class SporeSession(GameSession):

    def __init__(self, plugin, path):
        super().__init__(plugin, path)
        self.names: dict[int, str] = {}

    def name(self, value: int, hex_digits: int = 8) -> str:
        return self.names.get(value) or f"{value:#0{hex_digits + 2}x}"

    def raw(self, asset):
        if asset.ref[0] == "embedded":
            return b""
        return read_entry(asset.ref[1])

    def image(self, asset):
        tag = asset.ref[0]
        if tag == "embedded":
            _tag, entry, section = asset.ref
            return RW4(read_entry(entry)).raster(section)
        data = read_entry(asset.ref[1])
        ext = asset.ext
        if ext == "raster":
            return raster_image(data)
        if ext == "rw4":
            rw = RW4(data)
            rasters = rw.rasters()
            if not rasters:
                raise NotImplementedError("No texture in this RenderWare file")
            return rw.raster(rasters[0])
        return pil_image_from_bytes(data)

    def mesh(self, asset):
        return RW4(read_entry(asset.ref[1])).mesh()

    def materials(self, asset):
        entry = asset.ref[1]
        rw = RW4(read_entry(entry))
        rasters = rw.rasters()
        if not rasters:
            return []
        tex = Asset("texture", f"{asset.name} (texture)", ("embedded", asset.key, rasters[0]),
                    ref=("embedded", entry, rasters[0]), ext="png")
        return [Material(asset.name, [TextureRef("diffuse", tex.name, tex, ALBEDO)])]

    def text(self, asset):
        data = read_entry(asset.ref[1])
        if asset.ext == "prop":
            try:
                return prop_text(data, self.name)
            except (struct.error, ValueError, IndexError) as e:
                return f"(Couldn't read this property list: {e})\n\n" + data.hex(" ", 4)
        return data.removeprefix(b"\xef\xbb\xbf").decode("utf-8", "replace")

    def audio(self, asset):
        if asset.ext == "snr":
            from engines import extdecode
            return extdecode.to_wav(read_entry(asset.ref[1]), "snr"), "wav"
        return super().audio(asset)

    def describe(self, asset):
        rows = super().describe(asset)
        if asset.ref[0] == "entry":
            e = asset.ref[1]
            rows.append(("Key", f"group {e.group:#010x}, instance {e.instance:#010x}, type {e.type:#010x}"))
            rows.append(("Stored", f"{e.size:,} bytes" + (f" (RefPack, {e.mem_size:,} unpacked)" if e.compressed else "")))
        return rows


# --------------------------------------------------------------------------- plugin

class SporePlugin(EnginePlugin):
    id = "spore"
    name = "Spore (Maxis)"
    version = "1.0"
    author = "UniView"
    description = ("Spore's DBPF .package archives: RenderWare 4 models and textures, .raster/.png images, "
                   "property lists as text, locale texts and EA layer 3 sounds.")

    def _root(self, path: str) -> str | None:
        """The game folder: the path itself, or its parent when the Data / SporeBin folder was picked."""
        parent = os.path.dirname(path) if os.path.basename(path).lower() in ("data", "sporebin", "dataep1") else path
        for root in dict.fromkeys((path, parent)):
            if os.path.isfile(os.path.join(root, "Data", "Spore_Game.package")) or \
                    os.path.isfile(os.path.join(root, "SporeBin", "SporeApp.exe")):
                return root
        return None

    def detect(self, path):
        return 95 if self._root(path) else 0

    def game_info(self, path) -> GameInfo:
        root = self._root(path) or path
        ep1 = os.path.isdir(os.path.join(root, "DataEP1"))
        return {"engine_version": "Maxis DBPF", "detail": "Spore" + (" + Galactic Adventures" if ep1 else "")}

    def _packages(self, root: str) -> list[str]:
        found = []
        for folder in ("Data", "DataBP1", "DataEP1"):
            base = os.path.join(root, folder)
            if not os.path.isdir(base):
                continue
            mine = sorted(glob.glob(os.path.join(base, "*.package")), key=lambda p: os.path.basename(p).lower())
            mine.sort(key=lambda p: os.path.basename(p).lower().startswith("patch"))  # patches load last (win)
            found += mine + sorted(glob.glob(os.path.join(base, "Locale", "*", "*.package")))
        return found

    def count_files(self, path):
        root = self._root(path)
        return len(self._packages(root)) if root else 0

    def open(self, path, progress: Progress):
        root = self._root(path)
        if root is None:
            raise FileNotFoundError(f"No Spore Data folder found in:\n{path}")
        session = SporeSession(self, root)
        packages = self._packages(root)
        progress("Reading names", 0, len(packages) + 1)
        session.names = self._names(root)
        latest: dict[tuple, Entry] = {}
        for n, pkg in enumerate(packages):
            progress(f"Reading {os.path.basename(pkg)}", n + 1, len(packages) + 1)
            rel = os.path.relpath(pkg, root).replace("\\", "/")
            scope = rel.rsplit("/", 1)[0] if "/Locale/" in rel else ""  # each language keeps its own texts
            try:
                for e in read_index(pkg):
                    latest[(scope, e.type, e.group, e.instance)] = e
            except (OSError, ValueError, struct.error) as e:
                session.warnings.append(f"{rel}: {e}")
        session.file_count = len(packages)
        session.assets = self._assets(session, root, latest.values(), progress)
        return session

    def _assets(self, session, root, entries, progress) -> list[Asset]:
        entries = sorted(entries, key=lambda e: (e.package, e.offset))
        cache_path = os.path.join(cache_dir(), "spore_kinds.json")
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("version") != KINDS_CACHE_VERSION:
                cache = {}
        except (OSError, ValueError, AttributeError):
            cache = {}
        packages: dict[str, dict] = cache.setdefault("packages", {})
        cache["version"] = KINDS_CACHE_VERSION
        stamps = {}
        changed = False
        assets = []
        total = len(entries)
        for n, e in enumerate(entries):
            if n % 1000 == 0:
                progress(f"Listing files ({n:,}/{total:,})", n, total)
            ext, kind = TYPES.get(e.type, ("", "file"))
            if kind is None or not ext:
                if e.package not in stamps:
                    st = os.stat(e.package)
                    stamps[e.package] = f"{e.package}|{st.st_size}|{int(st.st_mtime)}"
                known = packages.setdefault(stamps[e.package], {})
                key = f"{e.type:x}:{e.group:x}:{e.instance:x}"
                if key in known:
                    ext, kind = known[key]
                else:
                    ext, kind = self._sniff(e)
                    known[key] = [ext, kind]
                    changed = True
            rel = os.path.relpath(e.package, root).replace("\\", "/")
            pkg = rel[:-len(".package")]
            name = session.name(e.instance)
            uid = f"{e.group:08x}!{e.instance:08x}.{e.type:08x}" + (f"@{pkg}" if "/Locale/" in rel else "")
            assets.append(Asset(kind, f"{name}.{ext}", uid, uid=uid, size=e.mem_size,
                                path=f"{pkg}/{session.name(e.group)}/{name}.{ext}", source=rel,
                                ref=("entry", e), ext=ext))
        if changed:
            for stamp in list(packages):
                path = stamp.rsplit("|", 2)[0]
                if path in stamps and stamps[path] != stamp:
                    del packages[stamp]  # from before the package changed (a game update)
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except OSError:
                pass
        return assets

    @staticmethod
    def _sniff(e: Entry) -> tuple[str, Kind]:
        """(extension, kind) of an entry whose type doesn't say enough: rw4 files, unknown types (text?)."""
        try:
            if e.type == TYPE_RW4:
                return "rw4", rw4_kind(e)
            text = _looks_like_text(read_entry(e, SNIFF_BYTES))
        except (OSError, ValueError, IndexError, struct.error):
            return TYPES.get(e.type, (f"{e.type:08x}",))[0], "file"
        return (text, "text") if text else (f"{e.type:08x}", "file")

    # -- names --

    def _names(self, root: str) -> dict[int, str]:
        """Hash -> name from the strings in SporeApp.exe (cached) plus any name lists (NAME_LISTS)."""
        names: dict[int, str] = {}
        exe = os.path.join(root, "SporeBin", "SporeApp.exe")
        if os.path.isfile(exe):
            stat = os.stat(exe)
            cache = os.path.join(cache_dir(), "spore_names.json")
            stamp = f"{NAMES_CACHE_VERSION}:{stat.st_size}:{int(stat.st_mtime)}"
            cached = None
            try:
                with open(cache, encoding="utf-8") as f:
                    cached = json.load(f)
            except (OSError, ValueError):
                pass
            if isinstance(cached, dict) and cached.get("stamp") == stamp:
                names.update((int(k), v) for k, v in cached["names"].items())
            else:
                with open(exe, "rb") as f:
                    data = f.read()
                words = {w.decode("latin-1") for w in _WORD.findall(data)}
                words |= {w.decode("latin-1") for w in _WORD.findall(data.replace(b"\0", b""))}
                for word in words:
                    names.setdefault(fnv1(word), word)
                try:
                    with open(cache, "w", encoding="utf-8") as f:
                        json.dump({"stamp": stamp, "names": names}, f)
                except OSError:
                    pass
        here = os.path.dirname(os.path.abspath(__file__))
        for folder in (os.path.join(here, NAME_LISTS), os.path.join(root, NAME_LISTS)):
            for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                names.update(read_name_list(path))
        return names


def read_name_list(path: str) -> dict[int, str]:
    """A name list: 'name' or 'name 0xHASH' per line (SporeModder's 'name	#hash' works too); '//' comments."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split("//", 1)[0].split()
            if not parts:
                continue
            value = None
            if len(parts) > 1:
                try:
                    value = int(parts[1].lstrip("#").removeprefix("0x"), 16)
                except ValueError:
                    pass
            out[value if value is not None else fnv1(parts[0])] = parts[0]
    return out


PLUGIN = SporePlugin()

