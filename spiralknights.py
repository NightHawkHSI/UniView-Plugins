"""Spiral Knights engine plugin for UniView (Three Rings' Clyde engine, Java).

The game keeps everything in three zip bundles under rsrc/ (intro-bundle.jar, full-rest-bundle.jar,
full-music-bundle.jar): PNG/JPG textures, OGG sounds and music, shaders, fonts, and ~7,500 .dat files.

.dat files are Clyde "binary exports" (com.threerings.export.BinaryExporter): a FACEAF0E header, then a zlib stream
of Java objects written field by field - class names the first time a class appears, ids for everything after.
ClydeReader below is a port of Clyde's BinaryImporter (Clyde (c) Three Rings Design, Inc., BSD licence,
https://github.com/threerings/clyde). Two kinds of value can only be read if you know their class:
enums (written as their name) and "Encodable" math types (Vector3f, Quaternion, Color4f ... as raw floats). The
enum classes are found by reading the class file headers in the game's own code/*.jar (ACC_ENUM flag).

What's shown:
  - models: ModelConfig .dat files - static meshes, articulated (rigged) models in their bind pose, static sets
    (the default variant), compound models (their parts placed together), derived and schemed models (the model
    they point at), with their textures
  - textures, sounds and music, fonts, shaders as text
  - every other .dat (animations, particle systems, UI layouts, configs) as a readable outline of its fields

Clyde is Z-up; models are turned Y-up for the viewer."""
from __future__ import annotations

import glob
import os
import re
import struct
import threading
import zipfile
import zlib
from typing import Any

import numpy as np

from engines.sdk import (
    ALBEDO, Asset, AssetStats, EnginePlugin, GameSession, Kind, Material, MeshData, TextureRef, kind_for_extension,
    pil_image_from_bytes, z_up_to_y_up,
)

BUNDLES = ("intro-bundle.jar", "full-rest-bundle.jar", "full-music-bundle.jar")  # later ones win on duplicates
IMAGE_EXTS = {"png", "jpg", "jpeg", "dds"}
TEXT_EXTS = {"frag", "vert", "tmpl", "txt", "xml", "properties"}
MODEL_IMPLS = {"ArticulatedConfig", "StaticConfig", "StaticSetConfig", "CompoundConfig", "ModelConfig$Derived",
               "ModelConfig$Schemed", "GeneratedStaticConfig", "ConditionalConfig"}
MAX_DEPTH = 6  # compound/derived models pointing at models pointing at models...


# ---------------------------------------------------------------- Clyde binary export reader

FINAL, INNER, COLLECTION, MAP, MULTI = 1, 2, 4, 8, 16
_PRIMS = ("boolean", "byte", "char", "double", "float", "int", "long", "short")  # bootstrap class ids 1..8
_PRIM_FMT = {"boolean": "?", "byte": "b", "char": "H", "double": "d", "float": "f", "int": "i", "long": "q",
             "short": "h"}
_BOXED = {"java.lang.Boolean": "boolean", "java.lang.Byte": "byte", "java.lang.Character": "char",
          "java.lang.Double": "double", "java.lang.Float": "float", "java.lang.Integer": "int",
          "java.lang.Long": "long", "java.lang.Short": "short"}
_ARRAY_FMT = {"[Z": "?", "[B": "b", "[C": "H", "[D": "d", "[F": "f", "[I": "i", "[J": "q", "[S": "h"}
_BUFFER_FMT = {"java.nio.ByteBuffer": "b", "java.nio.CharBuffer": "H", "java.nio.DoubleBuffer": "d",
               "java.nio.FloatBuffer": "f", "java.nio.IntBuffer": "i", "java.nio.LongBuffer": "q",
               "java.nio.ShortBuffer": "h"}
_STRINGS = {"java.lang.String", "java.lang.Class", "java.io.File"}
_ENCODABLE = {  # com.threerings.export.Encodable classes in the game: their decodeFromStream()
    "com.threerings.math.Matrix4f": "16f", "com.threerings.math.Matrix3f": "9f",
    "com.threerings.math.Quaternion": "4f", "com.threerings.math.Vector4f": "4f",
    "com.threerings.math.SphereCoords": "3f", "com.threerings.math.Vector2f": "2f",
    "com.threerings.math.Vector3f": "3f", "com.threerings.math.Plane": "4f",
    "com.threerings.opengl.renderer.Color4f": "4f", "com.threerings.tudey.util.Coord": "2i",
}
_ARRAY_COMPONENTS = {"Z": "boolean", "B": "byte", "C": "char", "D": "double", "F": "float", "I": "int",
                     "J": "long", "S": "short"}


class ClydeObject(dict):
    """An exported object: {field name: value}, plus its Java class name in .cls."""

    def __init__(self, cls: str) -> None:
        super().__init__()
        self.cls = cls

    @property
    def short(self) -> str:
        return self.cls.rsplit(".", 1)[-1]

    def __repr__(self) -> str:
        return f"<{self.short} {dict.__repr__(self)}>"

    def __hash__(self) -> int:  # type: ignore[override]  # usable as a map key (Clyde maps can have objects as keys)
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


class _Class:
    __slots__ = ("name", "flags", "comp")

    def __init__(self, name: str, flags: int, comp: _Class | None = None) -> None:
        self.name, self.flags, self.comp = name, flags, comp


class _Stop(Exception):
    """Raised by peek mode once the implementation class is known."""

    def __init__(self, name: str) -> None:
        self.name = name


class ClydeReader:
    """Reads one Clyde binary export (.dat). ClydeReader(data, enums).read() -> the root object."""

    def __init__(self, data: bytes, enums: frozenset[str] = frozenset()) -> None:
        if len(data) < 8:
            raise ValueError("too short for a Clyde export")
        magic, version, flags = struct.unpack_from(">IHH", data, 0)
        if magic != 0xFACEAF0E:
            raise ValueError(f"not a Clyde export (magic {magic:#x})")
        if version not in (0x1000, 0x1001, 0x1002):
            raise ValueError(f"unknown Clyde export version {version:#x}")
        self.b = zlib.decompress(data[8:]) if flags & 0x1000 else data[8:]
        self.p = 0
        self.enums = enums
        self.classic = version == 0x1000             # ids sized by the highest id so far
        self.int_lengths = version in (0x1000, 0x1001)  # lengths as 4-byte ints instead of varints
        self._by_name: dict[str, _Class] = {}
        self._classes = {i + 1: self._wrap(n, FINAL) for i, n in enumerate(_PRIMS)}
        self._object = self._wrap("java.lang.Object", 0)
        self._string = self._wrap("java.lang.String", FINAL)
        self._objects: dict[int, Any] = {0: None}
        self._highest = {"obj": 0, "cls": 0}
        self._fields: dict[str, dict] = {}
        self._peek = False

    def read(self) -> Any:
        return self._read(self._object)

    def peek(self) -> tuple[str, str]:
        """(root class, class of its 'implementation' field or '') without reading the rest."""
        self._peek = True
        try:
            root = self._read(self._object)
        except _Stop as stop:
            return self._root_class, stop.name
        return (root.cls if isinstance(root, ClydeObject) else type(root).__name__), ""

    # ---- low level
    def _unpack(self, fmt: str) -> tuple:
        v = struct.unpack_from(">" + fmt, self.b, self.p)
        self.p += struct.calcsize(">" + fmt)
        return v

    def _varint(self) -> int:
        ret = shift = 0
        b = self.b
        while True:
            c = b[self.p]
            self.p += 1
            ret |= (c & 0x7F) << shift
            if not c & 0x80:
                return ret
            shift += 7

    def _classic_id(self, highest: int) -> int:
        if highest < 255:
            return self._unpack("B")[0]
        if highest < 65535:
            return self._unpack("H")[0]
        return self._unpack("i")[0]

    def _id(self, kind: str) -> int:
        if not self.classic:
            return self._varint()
        v = self._classic_id(self._highest[kind])
        self._highest[kind] = max(self._highest[kind], v)
        return v

    def _length(self) -> int:
        return self._unpack("i")[0] if self.int_lengths else self._varint()

    def _utf(self) -> str:
        n = self._unpack("H")[0]
        s = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def _wrap(self, name: str, flags: int) -> _Class:
        c = self._by_name.get(name)
        if c is None:
            comp = None
            if name.startswith("["):  # arrays: final; the flags are the component's
                rest = name[1:]
                if rest.startswith("["):
                    comp = self._wrap(rest, flags)
                elif rest.startswith("L"):
                    comp = self._wrap(rest[1:-1], flags)
                else:
                    comp = self._wrap(_ARRAY_COMPONENTS[rest], FINAL)
                flags = FINAL
            c = self._by_name[name] = _Class(name, flags, comp)
        return c

    def _read_class(self) -> _Class:
        cid = self._id("cls")
        c = self._classes.get(cid)
        if c is None:
            name = self._utf()
            c = self._classes[cid] = self._wrap(name, self._unpack("b")[0])
        return c

    # ---- values
    def _read(self, cls: _Class) -> Any:
        if cls.name in _PRIM_FMT:
            return self._unpack(_PRIM_FMT[cls.name])[0]
        oid = self._id("obj")
        if oid in self._objects:
            return self._objects[oid]
        return self._value(cls, oid)

    def _value(self, cls: _Class, oid: int) -> Any:
        c = cls if cls.flags & FINAL else self._read_class()
        n = c.name
        if n in _PRIM_FMT or n in _BOXED:
            v: Any = self._unpack(_PRIM_FMT[_BOXED.get(n, n)])[0]
        elif n in _STRINGS or n in self.enums:
            v = self._utf()
        elif n in _ARRAY_FMT or n in _BUFFER_FMT:
            fmt = _ARRAY_FMT.get(n) or _BUFFER_FMT[n]
            count = self._unpack("i")[0]
            v = np.frombuffer(self.b, f">{fmt}", count, self.p).copy() if fmt != "?" else \
                [bool(x) for x in self.b[self.p:self.p + count]]
            self.p += count * struct.calcsize(fmt)
        elif n in _ENCODABLE:
            v = self._unpack(_ENCODABLE[n])
        else:
            return self._composite(c, n, oid)
        self._objects[oid] = v
        return v

    def _composite(self, c: _Class, n: str, oid: int) -> Any:
        if c.comp is not None:  # object array
            out: list = []
            self._objects[oid] = out
            for _ in range(self._length()):
                out.append(self._read(c.comp))
            return out
        if c.flags & INNER:
            self._read(self._object)  # the outer instance
        if "EnumSet" in n:
            self._read_class()  # the enum type
            out = []
            self._objects[oid] = out
            for _ in range(self._length()):
                out.append(self._read(self._object))
            return out
        if c.flags & COLLECTION:
            out = []
            self._objects[oid] = out
            multi = c.flags & MULTI
            for _ in range(self._length()):
                item = self._read(self._object)
                out.append((item, self._length()) if multi else item)
            return out
        if (c.flags & (MAP | MULTI)) == MAP:
            mapping: dict = {}
            self._objects[oid] = mapping
            for _ in range(self._length()):
                k = self._read(self._object)
                mapping[k if isinstance(k, (str, int, float, tuple, ClydeObject, type(None))) else repr(k)] = \
                    self._read(self._object)
            return mapping
        obj = ClydeObject(n)
        self._objects[oid] = obj
        depth_root = self._peek and len(self._objects) <= 2  # the root object, in peek mode
        if depth_root:
            self._root_class = n
        cdata = self._fields.setdefault(n, {"fields": {}, "highest": 0})
        for _ in range(self._length()):
            if self.classic:
                fid = self._classic_id(cdata["highest"])
                cdata["highest"] = max(cdata["highest"], fid)
            else:
                fid = self._varint()
            fd = cdata["fields"].get(fid)
            if fd is None:
                fd = cdata["fields"][fid] = (self._read(self._string), self._read_class())
            if depth_root and fd[0] == "implementation":
                if self._id("obj") in self._objects:
                    raise _Stop("")
                raise _Stop(fd[1].name if fd[1].flags & FINAL else self._read_class().name)
            obj[fd[0]] = self._read(fd[1])
        return obj


def scan_enums(code_dir: str) -> frozenset[str]:
    """Java names of the enum classes in the game's code jars (the ACC_ENUM flag in each class file)."""
    found = set()
    for jar in glob.glob(os.path.join(code_dir, "*.jar")):
        try:
            z = zipfile.ZipFile(jar)
        except (OSError, zipfile.BadZipFile):
            continue
        with z:
            for name in z.namelist():
                if not name.endswith(".class"):
                    continue
                try:
                    enum_name = _class_if_enum(z.read(name))
                except (struct.error, IndexError, KeyError, UnicodeDecodeError):
                    continue
                if enum_name:
                    found.add(enum_name)
    return frozenset(found)


def _class_if_enum(d: bytes) -> str | None:
    if d[:4] != b"\xca\xfe\xba\xbe":
        return None
    count = struct.unpack_from(">H", d, 8)[0]
    pos, i = 10, 1
    utf8: dict[int, bytes] = {}
    classes: dict[int, int] = {}
    while i < count:  # walk the constant pool to find where the access flags are
        tag = d[pos]
        pos += 1
        if tag == 1:
            ln = struct.unpack_from(">H", d, pos)[0]
            utf8[i] = d[pos + 2:pos + 2 + ln]
            pos += 2 + ln
        elif tag == 7:
            classes[i] = struct.unpack_from(">H", d, pos)[0]
            pos += 2
        elif tag in (5, 6):  # long/double take two slots
            pos += 8
            i += 1
        elif tag == 15:
            pos += 3
        elif tag in (8, 16, 19, 20):
            pos += 2
        else:
            pos += 4
        i += 1
    access, this = struct.unpack_from(">HH", d, pos)
    if not access & 0x4000:
        return None
    return utf8[classes[this]].decode("utf-8").replace("/", ".")


# ---------------------------------------------------------------- readable outline of any .dat

def outline(value: Any, depth: int = 0, seen: set[int] | None = None, max_items: int = 40) -> list[str]:
    """Lines describing an object tree, indented by depth; big number arrays are summarized."""
    seen = set() if seen is None else seen
    pad = "  " * depth
    if isinstance(value, ClydeObject):
        if id(value) in seen:
            return [f"{pad}(same {value.short} as above)"]
        seen.add(id(value))
        lines = [f"{pad}{value.short}"] if depth == 0 else []
        for k, v in value.items():
            if _simple(v):
                lines.append(f"{pad}  {k}: {_fmt(v)}")
            else:
                head = v.short if isinstance(v, ClydeObject) else f"[{len(v)}]" if isinstance(v, list) else \
                    f"{{{len(v)}}}" if isinstance(v, dict) else ""
                lines.append(f"{pad}  {k}: {head}")
                lines.extend(outline(v, depth + 2, seen, max_items) if not isinstance(v, ClydeObject) else
                             outline(v, depth + 1, seen, max_items)[0:])
        return lines
    if isinstance(value, list):
        lines = []
        for i, v in enumerate(value[:max_items]):
            if _simple(v):
                lines.append(f"{pad}[{i}] {_fmt(v)}")
            else:
                lines.append(f"{pad}[{i}] {v.short if isinstance(v, ClydeObject) else ''}")
                lines.extend(outline(v, depth + 1, seen, max_items))
        if len(value) > max_items:
            lines.append(f"{pad}... {len(value) - max_items} more")
        return lines
    if isinstance(value, dict):
        lines = []
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                lines.append(f"{pad}... {len(value) - max_items} more")
                break
            key = k.short if isinstance(k, ClydeObject) else repr(k) if not isinstance(k, str) else k
            if _simple(v):
                lines.append(f"{pad}{key}: {_fmt(v)}")
            else:
                lines.append(f"{pad}{key}: {v.short if isinstance(v, ClydeObject) else ''}")
                lines.extend(outline(v, depth + 1, seen, max_items))
        return lines
    return [pad + _fmt(value)]


def _simple(v: Any) -> bool:
    if isinstance(v, ClydeObject):
        return not v
    if isinstance(v, (list, dict)):
        return not v
    return True


def _fmt(v: Any) -> str:
    if isinstance(v, ClydeObject):
        return v.short + " (defaults)"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, np.ndarray):
        head = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v[:8].tolist())
        return f"{len(v)} values: {head}{' ...' if len(v) > 8 else ''}"
    if isinstance(v, (list, dict)):
        return "[]" if isinstance(v, list) else "{}"
    return str(v)


# ---------------------------------------------------------------- parameters

def clone(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Copy of an object tree's objects, lists and maps (leaves like arrays and strings are shared), so arguments
    can be applied without changing the cached original."""
    memo = {} if memo is None else memo
    if isinstance(value, (ClydeObject, list, dict)):
        found = memo.get(id(value))
        if found is not None:
            return found
        if isinstance(value, ClydeObject):
            out: Any = ClydeObject(value.cls)
            memo[id(value)] = out
            for k, v in value.items():
                out[k] = clone(v, memo)
        elif isinstance(value, list):
            out = []
            memo[id(value)] = out
            out.extend(clone(v, memo) for v in value)
        else:
            out = {}
            memo[id(value)] = out
            for k, v in value.items():
                out[k] = clone(v, memo)
        return out
    return value


def apply_arguments(root: ClydeObject, args: dict) -> ClydeObject:
    """Fill a config's parameters from a reference's arguments ({parameter name: value}), the way Clyde does for
    derived models and templates: each parameter names the field paths it sets."""
    params = {p.get("name"): p for p in root.get("parameters") or [] if isinstance(p, ClydeObject)}
    for name, value in args.items():
        param = params.get(name)
        if param is not None:
            _apply_parameter(root, param, value)
    return root


def _apply_parameter(root: ClydeObject, param: ClydeObject, value: Any) -> None:
    if param.short == "Parameter$Direct":
        for path in param.get("paths") or []:
            _set_path(root, str(path), value)
    elif param.short == "Parameter$Choice":  # value = option name; the option sets the choice's direct parameters
        option = next((o for o in param.get("options") or []
                       if isinstance(o, ClydeObject) and o.get("name") == value), None)
        if option is None:
            return
        directs = {d.get("name"): d for d in param.get("directs") or [] if isinstance(d, ClydeObject)}
        for name, v in (option.get("arguments") or {}).items():
            if name in directs:
                _apply_parameter(root, directs[name], v)


_PATH_TOKEN = re.compile(r'\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\["((?:[^"\\]|\\.)*)"\]')


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def _set_path(root: Any, path: str, value: Any) -> None:
    """Set the field a parameter path points at, e.g. implementation.models[0].model or
    implementation.actions[1].action.model["Sounder"]["Pitch adjust"] (["key"] on a reference = its argument)."""
    tokens: list[tuple[str, Any]] = []
    for m in _PATH_TOKEN.finditer(path):
        if m.group(1) is not None:
            tokens.append(("field", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("index", int(m.group(2))))
        else:
            tokens.append(("key", m.group(3).replace('\\"', '"')))
    node = root
    for i, (kind, tok) in enumerate(tokens):
        last = i == len(tokens) - 1
        if kind == "field":
            if not isinstance(node, ClydeObject):
                return
            name = tok if tok in node else _camel(tok)
            if last:
                node[name] = value
                return
            node = node.get(name)
        elif kind == "index":
            if not isinstance(node, list) or tok >= len(node):
                return
            if last:
                node[tok] = value
                return
            node = node[tok]
        else:
            if isinstance(node, ClydeObject) and node.short == "ConfigReference":
                args = node.get("arguments")
                if not isinstance(args, dict):
                    args = node["arguments"] = {}
                node = args
            if not isinstance(node, dict):
                return
            if last:
                node[tok] = value
                return
            node = node.get(tok)


# ---------------------------------------------------------------- geometry

def _matrix(transform: Any) -> np.ndarray:
    """A Clyde Transform3D as a 4x4 matrix (column vectors: p' = M @ [x, y, z, 1])."""
    m = np.eye(4)
    if not isinstance(transform, ClydeObject) or not transform:
        return m
    if "matrix" in transform:
        return np.array(transform["matrix"], dtype=np.float64).reshape(4, 4)
    x, y, z, w = transform.get("rotation", (0.0, 0.0, 0.0, 1.0))
    s = float(transform.get("scale", 1.0))
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m[:3, :3] = rot * s
    m[:3, 3] = transform.get("translation", (0.0, 0.0, 0.0))
    return m


def _array(config: Any, count_hint: int | None = None) -> np.ndarray | None:
    """(N, size) float array out of a ClientArrayConfig (interleaved floats: stride/offset in bytes)."""
    if not isinstance(config, ClydeObject):
        return None
    data = config.get("floatArray")
    if data is None or not len(data):
        return None
    data = np.asarray(data, dtype=np.float32)
    size = int(config.get("size", 4))
    stride = int(config.get("stride", 0)) // 4 or size
    offset = int(config.get("offset", 0)) // 4
    count = (len(data) - offset) // stride if stride else 0
    if count_hint is not None:
        count = min(count, count_hint)
    if count <= 0:
        return None
    idx = offset + np.arange(count)[:, None] * stride + np.arange(size)[None, :]
    idx = idx[(idx < len(data)).all(axis=1)]
    return data[idx]


def _triangles(indices: Any, mode: str, count: int) -> np.ndarray:
    idx = np.arange(count) if indices is None or not len(indices) else \
        (np.asarray(indices).astype(np.int64) & 0xFFFF if np.asarray(indices).dtype.itemsize == 2 else
         np.asarray(indices).astype(np.int64))
    if mode == "TRIANGLE_STRIP":
        tris = [(idx[i], idx[i + 1], idx[i + 2]) if i % 2 == 0 else (idx[i + 1], idx[i], idx[i + 2])
                for i in range(len(idx) - 2)]
        out = np.array(tris, dtype=np.int64).reshape(-1, 3)
        return out[(out[:, 0] != out[:, 1]) & (out[:, 1] != out[:, 2]) & (out[:, 0] != out[:, 2])]
    if mode == "TRIANGLE_FAN":
        return np.array([(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)], dtype=np.int64).reshape(-1, 3)
    if mode in ("TRIANGLES", ""):
        n = len(idx) // 3 * 3
        return idx[:n].reshape(-1, 3)
    return np.zeros((0, 3), dtype=np.int64)  # lines/points


class _Builder:
    """Collects the visible meshes of a model (and the models it points at) into one MeshData."""

    def __init__(self) -> None:
        self.points: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.uvs: list[np.ndarray] = []
        self.submeshes: list[np.ndarray] = []
        self.textures: list[str] = []  # resource path of each submesh's texture ("" = none)
        self.count = 0
        self.skipped = 0

    def add(self, geometry: Any, matrix: np.ndarray, texture: str) -> None:
        if not isinstance(geometry, ClydeObject):
            return
        pos = _array(geometry.get("vertexArray"))
        if pos is None or len(pos) == 0:
            self.skipped += 1
            return
        n = len(pos)
        pos = pos[:, :3].astype(np.float64)
        pos = pos @ matrix[:3, :3].T + matrix[:3, 3]
        normals = _array(geometry.get("normalArray"), n)
        if normals is not None and len(normals) == n:
            normals = normals[:, :3].astype(np.float64) @ np.linalg.inv(matrix[:3, :3]).T
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(lengths == 0, 1, lengths)
        else:
            normals = np.zeros((n, 3))
        tex = geometry.get("texCoordArrays") or []
        uv = _array(tex[0], n) if tex else None
        uv = uv[:, :2] if uv is not None and len(uv) == n else np.zeros((n, 2), np.float32)
        tris = _triangles(geometry.get("indices"), str(geometry.get("mode", "TRIANGLES")), n)
        tris = tris[(tris < n).all(axis=1)] if len(tris) else tris
        if not len(tris):
            self.skipped += 1
            return
        self.points.append(pos)
        self.normals.append(normals)
        self.uvs.append(uv)
        self.submeshes.append(tris + self.count)
        self.textures.append(texture)
        self.count += n

    def mesh(self, name: str) -> MeshData:
        if not self.points:
            raise ValueError("this model has no visible meshes (it may be an effect or a placeholder)")
        points = z_up_to_y_up(np.concatenate(self.points))
        normals = z_up_to_y_up(np.concatenate(self.normals))
        uvs = np.concatenate(self.uvs).astype(np.float32)
        uvs[:, 1] = 1.0 - uvs[:, 1]  # Clyde uploads images top row first: v=0 is the top
        return MeshData(points, self.submeshes, normals=normals, uvs={"UV0": uvs}, name=name,
                        skipped=self.skipped)


# ---------------------------------------------------------------- session

class SpiralKnightsSession(GameSession):
    def __init__(self, plugin: EnginePlugin, path: str) -> None:
        super().__init__(plugin, path)
        self.root = path
        self.entries: dict[str, tuple[str, zipfile.ZipInfo]] = {}  # lowercase path -> (bundle, entry)
        self.paths: dict[str, str] = {}  # lowercase path -> path as stored
        self.by_basename: dict[str, list[str]] = {}
        self.enums: frozenset[str] = frozenset()
        self._zips: dict[str, zipfile.ZipFile] = {}
        self._zip_lock = threading.Lock()
        self.asset_by_path: dict[str, Asset] = {}
        self._configs: dict[str, Any] = {}   # small caches on the session (not lru_cache: that would keep
        self._builds: dict[str, _Builder] = {}  # an unloaded game alive)

    # ---- files
    def read_path(self, path: str) -> bytes:
        key = path.replace("\\", "/").lstrip("/").lower()
        bundle, info = self.entries[key]
        with self._zip_lock:
            z = self._zips.get(bundle)
            if z is None:
                z = self._zips[bundle] = zipfile.ZipFile(bundle)
            return z.read(info)

    def raw(self, asset: Asset) -> bytes:
        return self.read_path(asset.ref)

    def close(self) -> None:
        with self._zip_lock:
            for z in self._zips.values():
                z.close()
            self._zips.clear()

    def config(self, path: str) -> Any:
        """The object stored in a .dat resource."""
        key = path.lower()
        value = self._configs.get(key)
        if value is None:
            value = ClydeReader(self.read_path(path), self.enums).read()
            if len(self._configs) >= 64:
                self._configs.pop(next(iter(self._configs)))
            self._configs[key] = value
        return value

    def resolve(self, name: str, near: str = "", source: str = "") -> str:
        """Resource path of a file referred to by `name` from the .dat at `near` ("" if not found): as a full
        path, next to the .dat, next to the model's source file, or else the closest file with that name."""
        if not name:
            return ""
        name = name.replace("\\", "/")
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".psd", ".tga", ".tif", ".tiff", ".bmp"):  # names of the artists' source files
            name = stem + ".png"
        candidates = [name]
        for base in (near, source):
            if base:
                candidates.append(os.path.dirname(base).replace("\\", "/") + "/" + name)
        for c in candidates:
            key = os.path.normpath(c).replace("\\", "/").lower()
            if key in self.entries:
                return self.paths[key]
        options = self.by_basename.get(os.path.basename(name).lower(), [])
        if not options:
            return ""
        here = os.path.dirname(near).lower().split("/")

        def shared(p: str) -> int:
            parts = os.path.dirname(p).lower().split("/")
            n = 0
            for a, b in zip(here, parts):
                if a != b:
                    break
                n += 1
            return n
        return self.paths[max(options, key=shared)]

    # ---- models
    def mesh(self, asset: Asset) -> MeshData:
        builder = self._build(asset.ref)
        return builder.mesh(os.path.splitext(os.path.basename(asset.ref))[0])

    def materials(self, asset: Asset) -> list[Material]:
        out = []
        for i, path in enumerate(self._build(asset.ref).textures):
            tex = self.asset_by_path.get(path.lower()) if path else None
            refs = [TextureRef("diffuse", tex.name, tex, ALBEDO)] if tex is not None else []
            out.append(Material(os.path.basename(path) or f"material {i}", refs))
        return out

    def _build(self, path: str) -> _Builder:
        """The model's meshes (cached: mesh(), materials() and stats() ask for the same model in a row)."""
        key = path.lower()
        builder = self._builds.get(key)
        if builder is None:
            builder = _Builder()
            self._add_model(builder, path, np.eye(4), 0, frozenset())
            if len(self._builds) >= 16:
                self._builds.pop(next(iter(self._builds)))
            self._builds[key] = builder
        return builder

    def _resolved(self, ref: Any) -> tuple[str, Any] | None:
        """(resource path, root config with the reference's arguments applied) for a model reference - a
        ConfigReference or a plain path - or None if it isn't a file (a config library entry) or is missing."""
        name = ref.get("name") if isinstance(ref, ClydeObject) else ref
        if not isinstance(name, str) or not name:
            return None
        key = name.replace("\\", "/").lower()
        if key not in self.entries:
            if key + ".dat" in self.entries:
                key += ".dat"
            else:
                return None
        root = self.config(self.paths[key])
        args = ref.get("arguments") if isinstance(ref, ClydeObject) else None
        if args and isinstance(root, ClydeObject):
            root = apply_arguments(clone(root), args)
        return self.paths[key], root

    def _add_model(self, b: _Builder, ref: Any, matrix: np.ndarray, depth: int, stack: frozenset[str]) -> None:
        resolved = self._resolved(ref)
        if resolved is None or depth > MAX_DEPTH or resolved[0].lower() in stack:
            return
        here, root = resolved
        stack = stack | {here.lower()}
        impl = root.get("implementation") if isinstance(root, ClydeObject) else None
        if not isinstance(impl, ClydeObject):
            return
        kind = impl.short
        if kind in ("StaticConfig", "GeneratedStaticConfig"):
            self._add_meshset(b, impl.get("meshes"), impl, here, matrix)
        elif kind == "StaticSetConfig":
            meshes = impl.get("meshes") or {}
            chosen = meshes.get(impl.get("model")) if isinstance(meshes, dict) else None
            if chosen is None and isinstance(meshes, dict) and meshes:
                chosen = next(iter(meshes.values()))
            self._add_meshset(b, chosen, impl, here, matrix)
        elif kind == "ArticulatedConfig":
            nodes: dict[str, np.ndarray] = {}
            self._add_node(b, impl.get("root"), impl, here, matrix, nodes)
            self._add_meshset(b, impl.get("skin"), impl, here, matrix)
            for att in impl.get("attachments") or []:  # parts hung on named nodes (armour, weapons, heads)
                if isinstance(att, ClydeObject):
                    at = nodes.get(att.get("node") or "", matrix)
                    self._add_model(b, att.get("model"), at @ _matrix(att.get("transform")), depth + 1, stack)
        elif kind == "CompoundConfig":
            for part in impl.get("models") or []:
                if isinstance(part, ClydeObject):
                    self._add_model(b, part.get("model"), matrix @ _matrix(part.get("transform")), depth + 1, stack)
        elif kind == "ModelConfig$Derived":
            self._add_model(b, impl.get("model"), matrix, depth + 1, stack)
        elif kind == "ModelConfig$Schemed":
            for part in impl.get("models") or []:  # the first that has a model: the highest detail
                ref = part.get("model") if isinstance(part, ClydeObject) else None
                if self._resolved(ref) is not None:
                    self._add_model(b, ref, matrix, depth + 1, stack)
                    break
        elif kind == "ConditionalConfig":  # picks a model by game state: show its default
            ref, transform = impl.get("defaultModel"), impl.get("defaultTransform")
            if self._resolved(ref) is None:
                for case in impl.get("cases") or []:
                    if isinstance(case, ClydeObject) and self._resolved(case.get("model")) is not None:
                        ref, transform = case.get("model"), case.get("transform")
                        break
            self._add_model(b, ref, matrix @ _matrix(transform), depth + 1, stack)

    def _add_node(self, b: _Builder, node: Any, impl: ClydeObject, here: str, parent: np.ndarray,
                  nodes: dict[str, np.ndarray]) -> None:
        if not isinstance(node, ClydeObject):
            return
        world = parent @ _matrix(node.get("transform"))
        if node.get("name"):
            nodes[node["name"]] = world
        visible = node.get("visible")
        if isinstance(visible, ClydeObject):
            self._add_visible(b, visible, impl, here, world)
        for child in node.get("children") or []:
            self._add_node(b, child, impl, here, world, nodes)

    def _add_meshset(self, b: _Builder, meshset: Any, impl: ClydeObject, here: str, matrix: np.ndarray) -> None:
        if isinstance(meshset, ClydeObject):
            for visible in meshset.get("visible") or []:
                self._add_visible(b, visible, impl, here, matrix)

    def _add_visible(self, b: _Builder, visible: Any, impl: ClydeObject, here: str, matrix: np.ndarray) -> None:
        if not isinstance(visible, ClydeObject):
            return
        b.add(visible.get("geometry"), matrix, self._texture(visible, impl, here))

    def _texture(self, visible: ClydeObject, impl: ClydeObject, here: str) -> str:
        """The texture a visible mesh is drawn with: its material mapping's Texture/File argument when there is
        one (that's what the game uses), else the texture named by the mesh."""
        name, tag = visible.get("texture") or "", visible.get("tag") or ""
        source = impl.get("source") or ""
        for mapping in impl.get("materialMappings") or []:
            if not isinstance(mapping, ClydeObject):
                continue
            if (mapping.get("texture") or "") != name or (mapping.get("tag") or "") != tag:
                continue
            file = _find_file_argument(mapping.get("material"))
            if file:
                found = self.resolve(file, here, source)
                if found:
                    return found
        return self.resolve(name, here, source)

    # ---- everything else
    def image(self, asset: Asset):
        return pil_image_from_bytes(self.raw(asset))

    def text(self, asset: Asset) -> str | bytes:
        if asset.ext == "dat":
            root = self.config(asset.ref)
            return "\n".join(outline(root)) + "\n"
        return super().text(asset)

    def describe(self, asset: Asset) -> list[tuple[str, str]]:
        rows = [("Bundle", os.path.basename(self.entries[asset.ref.lower()][0]))]
        if asset.ext == "dat":
            try:
                root = self.config(asset.ref)
            except Exception as e:
                return rows + [("Error", str(e))]
            if isinstance(root, ClydeObject):
                rows.append(("Config", root.short))
                impl = root.get("implementation")
                if isinstance(impl, ClydeObject):
                    rows.append(("Implementation", impl.short))
                    if impl.get("source"):
                        rows.append(("Made from", str(impl["source"])))
            if asset.kind == "model":
                try:
                    b = self._build(asset.ref)
                    rows.append(("Triangles", f"{sum(len(t) for t in b.submeshes):,}"))
                    rows.append(("Textures", ", ".join(sorted({os.path.basename(t) for t in b.textures if t}))
                                 or "none found"))
                except Exception as e:
                    rows.append(("Mesh", f"couldn't build: {e}"))
        return rows

    def stats(self, asset: Asset) -> AssetStats:
        if asset.kind == "model":
            try:
                b = self._build(asset.ref)
                tris = sum(len(t) for t in b.submeshes)
                return {"size": asset.size, "tris": tris, "verts": b.count, "info": f"{tris:,} tris",
                        "sort": tris}
            except Exception:
                return {"size": asset.size, "info": "?", "sort": -1}
        return {"size": asset.size, "info": "", "sort": asset.size or 0}


def _find_file_argument(ref: Any, depth: int = 0) -> str:
    """A material ConfigReference's texture file: the first image path among its (nested) arguments."""
    if not isinstance(ref, ClydeObject) or depth > 4:
        return ""
    args = ref.get("arguments") or {}
    if isinstance(args, dict):
        for k, v in args.items():
            if isinstance(v, str) and os.path.splitext(v)[1].lower().lstrip(".") in IMAGE_EXTS:
                return v
            if isinstance(v, ClydeObject):
                found = _find_file_argument(v, depth + 1)
                if found:
                    return found
    return ""


# ---------------------------------------------------------------- plugin

class SpiralKnightsPlugin(EnginePlugin):
    id = "spiralknights"
    name = "Spiral Knights (Clyde)"
    version = "1.0"
    author = "UniView"
    description = "Spiral Knights: models, textures, sounds and music from rsrc/*.jar (Three Rings' Clyde engine)."

    @staticmethod
    def _root(path: str) -> str | None:
        path = os.path.normpath(path)
        for p in (path, os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
            if os.path.isfile(os.path.join(p, "rsrc", "full-rest-bundle.jar")) or \
                    os.path.isfile(os.path.join(p, "rsrc", "intro-bundle.jar")):
                return p
        return None

    def detect(self, path: str) -> int:
        root = self._root(path)
        if root is None:
            return 0
        return 95 if os.path.isfile(os.path.join(root, "code", "projectx-pcode.jar")) else 60

    def game_info(self, path: str):
        return {"engine_version": "Clyde", "detail": "Spiral Knights"}

    def count_files(self, path: str) -> int:
        root = self._root(path) or path
        total = 0
        for name in BUNDLES:
            try:
                with zipfile.ZipFile(os.path.join(root, "rsrc", name)) as z:
                    total += sum(1 for i in z.infolist() if not i.is_dir())
            except (OSError, zipfile.BadZipFile):
                pass
        return total

    def open(self, path: str, progress) -> SpiralKnightsSession:
        root = self._root(path)
        if root is None:
            raise FileNotFoundError(f"No Spiral Knights rsrc/*.jar bundles found in:\n{path}")
        session = SpiralKnightsSession(self, root)
        progress("Reading the game's class list ...")
        session.enums = scan_enums(os.path.join(root, "code"))
        bundles = [os.path.join(root, "rsrc", n) for n in BUNDLES if os.path.isfile(os.path.join(root, "rsrc", n))]
        progress.files((os.path.basename(b), os.path.getsize(b), b) for b in bundles)
        for i, bundle in enumerate(bundles):
            progress.file(i)
            progress(f"Listing {os.path.basename(bundle)} ...", i, len(bundles))
            with zipfile.ZipFile(bundle) as z:
                for info in z.infolist():
                    if info.is_dir() or info.filename.startswith("META-INF/"):
                        continue
                    key = info.filename.lower()
                    session.entries[key] = (bundle, info)
                    session.paths[key] = info.filename
        progress.file(len(bundles))
        for key, name in session.paths.items():
            session.by_basename.setdefault(os.path.basename(key), []).append(key)

        found: dict[str, int] = {}
        kind: Kind
        keys = sorted(session.paths)
        dats = [k for k in keys if k.endswith(".dat")]
        done = 0
        for key in keys:
            name = session.paths[key]
            ext = os.path.splitext(key)[1].lstrip(".")
            info = session.entries[key][1]
            stem = name[:-len(ext) - 1] if ext else name
            if ext == "dat":
                done += 1
                if done % 250 == 0:
                    progress(f"Sorting configs {done:,}/{len(dats):,} ...", done, len(dats))
                    progress.found(found)
                try:
                    root_cls, impl_cls = ClydeReader(session.read_path(name), session.enums).peek()
                except Exception:
                    root_cls, impl_cls = "", ""
                short_root, short_impl = root_cls.rsplit(".", 1)[-1], impl_cls.rsplit(".", 1)[-1]
                if short_root == "ModelConfig" and short_impl in MODEL_IMPLS:
                    kind, label = "model", stem
                else:
                    kind = "data"
                    what = short_root.replace("Config", "") or "Data"
                    if short_impl and short_root == "ModelConfig":
                        what = short_impl.replace("Config", "").replace("ModelConfig$", "")
                    label = f"{what}: {stem}"
            elif ext in IMAGE_EXTS:
                kind, label = "texture", stem
            elif ext == "ogg":
                kind, label = "audio", name
            elif ext in ("ttf", "otf"):
                kind, label = "font", name
            elif ext in TEXT_EXTS:
                kind, label = "text", name
            else:
                kind, label = kind_for_extension(ext), name
            asset = Asset(kind=kind, name=label, key=key, uid="sk:" + key, size=info.file_size, path=name,
                          source=os.path.dirname(name), ref=name, ext=ext)
            session.assets.append(asset)
            session.asset_by_path[key] = asset
            found[kind] = found.get(kind, 0) + 1
        progress.found(found)
        session.file_count = len(bundles)
        return session


PLUGIN = SpiralKnightsPlugin()
