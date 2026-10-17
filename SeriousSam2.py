"""Serious Sam 2 (Croteam, Serious Engine 2).

The game stores everything in .gro archives next to the exe (All_PC_01.gro,
All_PC_02.gro, All_PC_03.gro, the Movies_PC_*.gro and the patch archives). A .gro
is simply a ZIP archive with a renamed extension -- the engine reads them with
its own ZIP reader -- so this plugin opens them with Python's zipfile module.

Textures are .tex files (TVER/TDAT): a small header, a mip chain, and pixels in
one of several formats (DXT1/3/5, 32-bit BGRA, 24-bit BGR, 8-bit gray). Models
are .mdl files (Serious Engine 2's binary mesh format) which this plugin lists
but does not decode yet. Loose .png/.tga/.jpg images and .txt/.ini/.xml files
next to the archives are listed too.

What's shown: every texture in the archives (with a size column), every model,
every sound (listed as raw), and loose images / text files. Code, worlds and
animations are not read.
"""
from __future__ import annotations

import io
import logging
import os
import struct
import threading
import zipfile
from collections import OrderedDict

from engines.sdk import Asset, AssetStats, EnginePlugin, GameInfo, GameSession, Kind, \
    kind_for_extension, pil_image_from_bytes

log = logging.getLogger("viewer.serious")

LOOSE_IMAGES = {"png", "tga", "jpg", "jpeg", "bmp", "gif"}
LOOSE_TEXTS = {"txt", "ini", "xml", "json", "csv", "md", "log"}
TEX_MAGIC = b"TVER"
TEX_VERSION_3 = 3
TEX_VERSION_4 = 4
ZIP_CACHE = 8  # open .gro archives kept in memory (they're big; the OS page cache handles the rest)

# D3D formats seen in SE2 .tex files (low byte of the header flags)
FMT_DXT1 = 0x01
FMT_DXT3 = 0x02
FMT_DXT5 = 0x03
FMT_BGRA8888 = 0x04
FMT_BGR888 = 0x05
FMT_L8 = 0x06
FMT_A8 = 0x07
FMT_BGRX8888 = 0x08


class _LRU(OrderedDict):
    """Tiny LRU so browsing a long texture list doesn't thrash the open-archive cache (only a
    few large .gro files are kept open at once)."""

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    def get_lru(self, key):
        v = self.get(key)
        if v is not None:
            self.move_to_end(key)
        return v

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.cap:
            self.popitem(last=False)


# ---------------------------------------------------------------- .tex textures

def _tex_header(data: bytes) -> dict:
    if data[:4] != TEX_MAGIC:
        raise ValueError("not a Serious Engine .tex file")
    version, = struct.unpack_from("<I", data, 4)
    if version not in (TEX_VERSION_3, TEX_VERSION_4):
        raise ValueError(f"unsupported .tex version {version}")
    pos = 8
    # "TDAT" usually sits at offset 8 (v3) or 12 (v4); skip past whichever marker is present.
    for magic in (b"TDAT", b"TFNM"):
        if data[pos:pos + 4] == magic:
            pos += 4
            break
    flags, width, height = struct.unpack_from("<III", data, pos)
    pos += 12
    depth = 1
    if version == TEX_VERSION_4:
        depth, = struct.unpack_from("<I", data, pos)
        pos += 4
    mip_count, = struct.unpack_from("<I", data, pos)
    pos += 4
    return {"version": version, "flags": flags, "width": width, "height": height,
            "depth": max(1, depth), "mips": max(1, mip_count), "data_offset": pos}


def tex_image(data: bytes):
    """PIL image of a Serious Engine .tex file (mip 0 only; DXT1/3/5, BGRA/BGR, gray)."""
    from PIL import Image
    hdr = _tex_header(data)
    w, h = hdr["width"], hdr["height"]
    if w == 0 or h == 0:
        raise ValueError("empty texture")
    pos = hdr["data_offset"]
    fmt = hdr["flags"] & 0xFF
    if fmt in (FMT_DXT1, FMT_DXT3, FMT_DXT5):
        block_bytes = {FMT_DXT1: 8, FMT_DXT3: 16, FMT_DXT5: 16}[fmt]
        size = ((w + 3) // 4) * ((h + 3) // 4) * block_bytes
        if pos + size > len(data):
            raise ValueError("truncated DXT mip")
        pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
        bcn = {FMT_DXT1: 1, FMT_DXT3: 2, FMT_DXT5: 3}[fmt]
        img = Image.frombytes("RGBA", (pw, ph), data[pos:pos + size], "bcn", bcn)
        return img.crop((0, 0, w, h)) if (pw, ph) != (w, h) else img
    if fmt in (FMT_BGRA8888, FMT_BGRX8888):
        size = w * h * 4
        if pos + size > len(data):
            raise ValueError("truncated 32-bit mip")
        img = Image.frombytes("RGBA", (w, h), data[pos:pos + size], "raw", "BGRA")
        return img.convert("RGB") if fmt == FMT_BGRX8888 else img
    if fmt == FMT_BGR888:
        size = w * h * 3
        if pos + size > len(data):
            raise ValueError("truncated 24-bit mip")
        return Image.frombytes("RGB", (w, h), data[pos:pos + size], "raw", "BGR")
    if fmt in (FMT_L8, FMT_A8):
        size = w * h
        if pos + size > len(data):
            raise ValueError("truncated 8-bit mip")
        return Image.frombytes("L", (w, h), data[pos:pos + size])
    raise NotImplementedError(f"texture format {fmt:#x} isn't supported yet")


def tex_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a .tex header without decoding the pixels."""
    hdr = _tex_header(data)
    return hdr["width"], hdr["height"]


# ---------------------------------------------------------------- session

class SeriousSession(GameSession):
    def __init__(self, plugin: EnginePlugin, path: str) -> None:
        super().__init__(plugin, path)
        self._zip_lock = threading.Lock()
        self._zips: _LRU = _LRU(ZIP_CACHE)
        self._by_key: dict | None = None

    def close(self) -> None:
        with self._zip_lock:
            for z in self._zips.values():
                z.close()
            self._zips.clear()

    def _zip(self, path: str) -> zipfile.ZipFile:
        with self._zip_lock:
            z = self._zips.get_lru(path)
            if z is None:
                z = zipfile.ZipFile(path)
                self._zips.put(path, z)
            return z

    def raw(self, asset: Asset) -> bytes:
        tag = asset.ref[0]
        if tag == "gro":
            _tag, gro_path, member = asset.ref
            return self._zip(gro_path).read(member)
        if tag == "loose":
            with open(asset.ref[1], "rb") as f:
                return f.read()
        raise ValueError(f"unknown asset {asset!r}")

    def image(self, asset: Asset):
        if asset.ref[0] == "gro" and asset.ext == "tex":
            return tex_image(self.raw(asset))
        return pil_image_from_bytes(self.raw(asset))

    def text(self, asset: Asset) -> str:
        return self.raw(asset).removeprefix(b"\xef\xbb\xbf").decode("utf-8", "replace")

    def stats(self, asset: Asset) -> AssetStats:
        stats: AssetStats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        try:
            if asset.ext == "tex" and asset.ref[0] == "gro":
                w, h = tex_size(self.raw(asset))
                stats.update(w=w, h=h, info=f"{w}\u00d7{h}", sort=w * h)
            elif asset.ext in LOOSE_IMAGES and asset.ref[0] == "loose":
                from PIL import Image
                w, h = Image.open(io.BytesIO(self.raw(asset))).size
                stats.update(w=w, h=h, info=f"{w}\u00d7{h}", sort=w * h)
        except Exception as e:
            log.debug("No stats for %s: %s", asset.name, e)
        return stats

    def describe(self, asset: Asset) -> list[tuple[str, str]]:
        rows = super().describe(asset)
        if asset.ref[0] == "gro":
            _tag, gro_path, member = asset.ref
            rows.append(("Archive", os.path.basename(gro_path)))
            rows.append(("Stored as", member))
        elif asset.ref[0] == "loose":
            rows.append(("File", asset.ref[1]))
        return rows


# ---------------------------------------------------------------- plugin

class SeriousPlugin(EnginePlugin):
    id = "serious"
    name = "Serious Sam 2"
    version = "1.1"
    author = "UniView"
    description = ("Serious Sam 2 (Serious Engine 2): .gro archives (ZIP containers), .tex textures, "
                   ".mdl models and loose files next to the game.")

    @staticmethod
    def _root(path: str) -> str | None:
        """The game folder: the path itself, its parent, or the grandparent (so picking a subfolder works)."""
        path = os.path.normpath(path)
        for p in (path, os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
            if os.path.isdir(p) and any(
                    os.path.isfile(os.path.join(p, f)) for f in ("All_PC_01.gro", "Sam2.exe", "SeriousSam2.exe")):
                return p
        return None

    def detect(self, path: str) -> int:
        root = self._root(path)
        if root is None:
            return 0
        if os.path.isfile(os.path.join(root, "Sam2.exe")) or os.path.isfile(os.path.join(root, "SeriousSam2.exe")):
            return 95
        return 80 if any(f.lower().endswith(".gro") for f in os.listdir(root)) else 0

    def game_info(self, path: str) -> GameInfo:
        root = self._root(path) or path
        detail = "Serious Sam 2"
        try:
            gros = [f for f in os.listdir(root) if f.lower().endswith(".gro")]
            if gros:
                detail += f" \u00b7 {len(gros)} archive(s)"
        except OSError:
            pass
        return {"engine_version": "Serious Engine 2", "detail": detail}

    def _gro_files(self, root: str) -> list[str]:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            return []
        return [os.path.join(root, n) for n in names if n.lower().endswith(".gro")]

    def count_files(self, path: str) -> int:
        root = self._root(path) or path
        return len(self._gro_files(root))

    def open(self, path: str, progress) -> SeriousSession:
        root = self._root(path)
        if root is None:
            raise FileNotFoundError(f"No Serious Sam 2 data (a .gro archive or Sam2.exe) found in:\n{path}")
        session = SeriousSession(self, root)
        gros = self._gro_files(root)
        if not gros:
            raise FileNotFoundError(f"No .gro archives found in:\n{root}")
        progress.files([(os.path.basename(g), os.path.getsize(g), g) for g in gros])
        found: dict[str, int] = {}

        def add(asset: Asset) -> None:
            session.assets.append(asset)
            found[asset.kind] = found.get(asset.kind, 0) + 1

        for n, gro_path in enumerate(gros, 1):
            progress.file(n - 1)
            progress(f"Reading {os.path.basename(gro_path)} ({n}/{len(gros)})", n - 1, len(gros))
            try:
                z = session._zip(gro_path)
            except (OSError, zipfile.BadZipFile) as e:
                session.warnings.append(f"{os.path.basename(gro_path)}: {e}")
                continue
            for member in z.namelist():
                if member.endswith("/"):
                    continue  # directory entry
                low = member.lower()
                ext = low.rsplit(".", 1)[-1] if "." in low else ""
                kind: Kind
                if ext == "tex":
                    kind = "texture"
                elif ext == "mdl":
                    kind = "model"
                elif ext in ("wav", "ogg", "mp3"):
                    kind = "audio"
                elif ext in ("txt", "ini", "xml", "json", "csv"):
                    kind = "text"
                else:
                    kind = kind_for_extension(ext)
                name = member.replace("\\", "/")
                asset = Asset(kind, name, ("gro", gro_path, member),
                              uid=f"ss2:{os.path.basename(gro_path)}:{name}",
                              size=z.getinfo(member).file_size, path=name,
                              source=os.path.basename(gro_path), ref=("gro", gro_path, member),
                              ext=ext or "bin")
                add(asset)
            progress.found(found)
        # Loose files next to the archives.
        progress("Looking for loose files ...", len(gros), len(gros) + 1)
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext in LOOSE_IMAGES:
                    kind = "texture"
                elif ext in LOOSE_TEXTS:
                    kind = "text"
                else:
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root).replace("\\", "/")
                add(Asset(kind=kind, name=f"files/{rel}", key=("loose", rel.lower()),
                          uid=f"ss2:loose:{rel.lower()}", size=os.path.getsize(full), path=rel,
                          source="loose file", ref=("loose", full), ext=ext))
        session.file_count = len(gros)
        if not session.assets:
            raise RuntimeError("Nothing could be listed from this game's archives.")
        return session


PLUGIN = SeriousPlugin()