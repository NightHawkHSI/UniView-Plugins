"""Fallout Tactics: Brotherhood of Steel (Micro Forté, 2001) plugin for UniView.

The game keeps its art in .bos archives under the game's core/ folder:

  tiles_0.bos       map tiles (.til) and their embedded images
  entities_0.bos    entity definitions (.ent) for every weapon, armour, NPC, ...
  sprites.bos       character / vehicle / effect sprites (.spr)
  mis-main_0.bos    mission files (.mis)
  locale.bos        text (items, dialogue, UI) under locale/
  movie/*.bik       Bink movies (loose, next to core/)

A .bos is simply a ZIP archive with a renamed extension -- the game's own readme
and every modding guide confirm it, and WinZip / 7-Zip open them directly -- so
this plugin reads them with Python's zipfile module.

Inside the archives:

  .zar   standalone images (also embedded in .til and .spr): a 6-byte signature,
         a type byte, width/height, an optional palette and RLE-compressed pixels
  .til   map tiles, each holding one or more embedded .zar images
  .spr   sprite animations: a header with sequences and animations, then
         per-frame image blocks ('<spranim_img>'), uncompressed or zlib-packed
  .ent   binary entity definitions, listed as text (the format is documented)
  .mis   mission files, listed as files
  .bik   Bink videos, listed as files
  .txt / .ini / .cfg  plain text

What's shown: every image in the archives, the sprite animations (best effort:
the palette for raw .spr frames is not stored in the file, so those come out
grayscale), the tile images, all texts, and every other file classified by its
extension. Loose .bik / .png / .txt files next to the game are listed too.
"""
from __future__ import annotations

import io
import logging
import os
import struct
import threading
import zipfile
import zlib
from collections import OrderedDict

import numpy as np

from engines.sdk import (Asset, AssetStats, EnginePlugin, GameInfo, GameSession, Kind,
                         pil_image_from_bytes)

log = logging.getLogger("viewer.fot")

LOOSE_IMAGES = {"png", "jpg", "jpeg", "bmp", "gif", "tga"}
LOOSE_TEXTS = {"txt", "ini", "cfg", "xml", "json", "csv", "log", "md"}
ZAR_SIG = b"<zar>"
ZAR_TYPE = 0x33
ZAR_TYPE_PAL = 0x34
SPR_SIG = b"<sprite>"
SPRANIM_SIG = b"<spranim>"
SPRANIM_IMG_SIG = b"<spranim_img>"
SPR_VERSION = b"4"
ZIP_CACHE = 8  # archives kept open at once (the big ones are hundreds of MB)

# Extension -> Kind. Anything not listed becomes a "file".
KINDS: dict[str, Kind] = {
    "zar": "texture", "til": "texture", "spr": "animation",
    "bik": "file", "mis": "file", "ent": "text", "txt": "text", "ini": "text",
    "cfg": "text", "wav": "audio", "ogg": "audio", "mp3": "audio",
    "png": "texture", "jpg": "texture", "jpeg": "texture", "bmp": "texture",
    "gif": "texture", "tga": "texture", "dds": "texture",
}


class _LRU(OrderedDict):
    """Tiny LRU so scrolling a long archive listing keeps the archive you just used open."""

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


# ---------------------------------------------------------------- ZAR images

def _rle_pixels(data: bytes, pos: int, width: int, height: int, palette, default_color: int):
    """Decode a ZAR image's RLE blocks into an RGBA pixel list. Each block starts with a byte whose
    low two bits are the command and whose high six bits are the run length. Blocks never cross a
    scanline; any overflow is read but discarded."""
    from PIL import Image
    pixels = [(0, 0, 0, 0)] * (width * height)
    x = y = 0
    end = len(data)
    while pos < end and y < height:
        hdr = data[pos]
        pos += 1
        cmd = hdr & 3
        run = hdr >> 2
        if run == 0:
            continue
        if cmd == 0:
            for i in range(run):
                if x + i < width and y < height:
                    pixels[y * width + x + i] = (0, 0, 0, 0)
        elif cmd == 1:
            for i in range(run):
                if pos >= end:
                    break
                idx = data[pos]
                pos += 1
                if x + i < width and y < height and palette and idx < len(palette):
                    r, g, b, _ = palette[idx]
                    pixels[y * width + x + i] = (r, g, b, 255)
        elif cmd == 2:
            for i in range(run):
                if pos + 1 >= end:
                    break
                idx, alpha = data[pos], data[pos + 1]
                pos += 2
                if x + i < width and y < height and palette and idx < len(palette):
                    r, g, b, _ = palette[idx]
                    pixels[y * width + x + i] = (r, g, b, alpha)
        elif cmd == 3:
            for i in range(run):
                if pos >= end:
                    break
                alpha = data[pos]
                pos += 1
                if x + i < width and y < height and palette:
                    r, g, b, _ = palette[min(default_color, len(palette) - 1)]
                    pixels[y * width + x + i] = (r, g, b, alpha)
        x += run
        while x >= width:
            x -= width
            y += 1
    return Image.fromarray(np.array(pixels, np.uint8).reshape(height, width, 4), "RGBA")


def zar_image(data: bytes, start: int = 0):
    """PIL image of a ZAR image at `start` in `data`. Standalone .zar files carry their own palette;
    images embedded in .til / .spr may not (those come out using the palette index as greyscale)."""
    if data[start:start + 5] != ZAR_SIG:
        raise ValueError("not a ZAR image")
    zar_type = data[start + 6]
    if zar_type not in (ZAR_TYPE, ZAR_TYPE_PAL):
        raise ValueError(f"unknown ZAR type {zar_type:#x}")
    width, height = struct.unpack_from("<II", data, start + 8)
    if not width or not height or width > 8192 or height > 8192:
        raise ValueError(f"implausible ZAR size {width}x{height}")
    pos = start + 16
    palette = None
    default_color = 0
    if data[pos]:  # palette present
        pos += 1
        color_count, = struct.unpack_from("<I", data, pos)
        pos += 4
        if color_count > 256:
            raise ValueError(f"implausible palette size {color_count}")
        palette = []
        for _ in range(color_count):
            b, g, r, a = data[pos:pos + 4]
            palette.append((r, g, b, a))
            pos += 4
        if zar_type == ZAR_TYPE_PAL:
            default_color = data[pos]
            pos += 1
    else:
        pos += 1
    data_length, = struct.unpack_from("<I", data, pos)
    pos += 4
    if not palette:
        # No palette in the file: show the indices as greys so the shape is visible.
        palette = [(i, i, i, 255) for i in range(256)]
    return _rle_pixels(data, pos, width, height, palette, default_color)


def find_zar(data: bytes, start: int = 0) -> int:
    """Offset of the first ZAR signature at or after `start`, or -1."""
    at = data.find(ZAR_SIG, start)
    while at >= 0:
        if at + 17 < len(data) and data[at + 6] in (ZAR_TYPE, ZAR_TYPE_PAL):
            return at
        at = data.find(ZAR_SIG, at + 1)
    return -1


def zar_size(data: bytes, start: int) -> tuple[int, int]:
    return struct.unpack_from("<II", data, start + 8)


# ---------------------------------------------------------------- .spr sprites

def spr_info(data: bytes) -> dict:
    """Parse a .spr sprite header: name, centering point, sequences and animation collections.

    Header (from the TeamX spr2gif documentation):
      Uint8[11]  '<sprite>\\0' + '4\\0'
      Uint8[3]   unknown
      Uint32 x, y  centering coordinates
      Uint8[3]   unknown
      Uint32 seq_count
      SEQ_HDR[seq_count]
      Uint32 anm_count
      ANM_HDR[anm_count]
    """
    if data[:8] != SPR_SIG:
        raise ValueError("not a .spr sprite")
    pos = 11 + 3
    cx, cy = struct.unpack_from("<II", data, pos)
    pos += 8 + 3
    seq_count, = struct.unpack_from("<I", data, pos)
    pos += 4
    sequences = []
    for _ in range(seq_count):
        itm_count, = struct.unpack_from("<I", data, pos)
        pos += 4 + itm_count * 2 + itm_count * 4
        length, = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos:pos + length].decode("latin-1", "replace")
        pos += length
        anim_idx, = struct.unpack_from("<H", data, pos)
        pos += 2
        sequences.append({"name": name, "anim": anim_idx})
    anm_count, = struct.unpack_from("<I", data, pos)
    pos += 4
    animations = []
    for _ in range(anm_count):
        pos += 12  # '<spranim>\0' + '1\0'
        img_offset, = struct.unpack_from("<I", data, pos)
        pos += 4
        length, = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos:pos + length].decode("latin-1", "replace")
        pos += length
        frm_count, dir_count = struct.unpack_from("<II", data, pos)
        pos += 8
        pos += frm_count * dir_count * 16  # RECT[frm_count * dir_count]
        animations.append({"name": name, "frames": frm_count, "directions": dir_count,
                           "offset": img_offset})
    return {"center": (cx, cy), "sequences": sequences, "animations": animations}


def spr_frame(data: bytes, offset: int):
    """One frame image from a .spr: the pixel block at `offset` ('<spranim_img>' header, then
    '1' for raw or '2' for zlib). Raw frames are 8bpp indices; the game's palette is not in the
    file, so they are shown as greyscale."""
    from PIL import Image
    if data[offset:offset + 13] != SPRANIM_IMG_SIG:
        raise ValueError("no image block at this offset")
    pos = offset + 14
    kind = data[pos:pos + 2]
    pos += 2
    if kind == b"2\0":
        decompressed_size, = struct.unpack_from("<I", data, pos)
        pos += 4
        # The compressed size is not stored; the next block's signature marks the end.
        next_at = data.find(SPRANIM_IMG_SIG, pos)
        packed = data[pos:next_at] if next_at > 0 else data[pos:]
        raw = zlib.decompress(packed, 15, decompressed_size)
    elif kind == b"1\0":
        raw = data[pos:]
    else:
        raise ValueError(f"unknown .spr frame kind {kind!r}")
    if not raw:
        raise ValueError("empty frame")
    # Raw frames are 8bpp; the true size comes from the bounding box, but without the image
    # table we use a square-ish guess and let the caller clip it.
    side = int(len(raw) ** 0.5)
    if side * side == len(raw):
        arr = np.frombuffer(raw, np.uint8, side * side).reshape(side, side)
    else:
        arr = np.frombuffer(raw, np.uint8, len(raw)).reshape(1, -1)
    return Image.fromarray(arr, "L").convert("RGBA")


# ---------------------------------------------------------------- session

class FotSession(GameSession):
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
        if tag == "bos":
            _t, bos_path, member = asset.ref
            return self._zip(bos_path).read(member)
        if tag == "loose":
            with open(asset.ref[1], "rb") as f:
                return f.read()
        raise ValueError(f"unknown asset {asset!r}")

    def _data(self, asset: Asset) -> bytes:
        return self.raw(asset)

    def image(self, asset: Asset):
        tag = asset.ref[0]
        if tag == "bos" and asset.ext == "zar":
            return zar_image(self.raw(asset))
        if tag == "bos" and asset.ext == "til":
            data = self.raw(asset)
            at = find_zar(data)
            if at < 0:
                raise ValueError("this .til has no embedded ZAR image")
            return zar_image(data, at)
        if tag == "bos" and asset.ext == "spr":
            return self._spr_frame(asset, 0)
        if tag == "sprframe":
            _t, spr_asset, index = asset.ref
            return self._spr_frame(spr_asset, index)
        return pil_image_from_bytes(self.raw(asset))

    def _spr_frame(self, spr_asset: Asset, index: int):
        data = self.raw(spr_asset)
        info = spr_info(data)
        animations = info["animations"]
        if not animations:
            raise ValueError("this .spr has no animations")
        anim = animations[min(index, len(animations) - 1)]
        return spr_frame(data, anim["offset"])

    # ---- animation API (the SDK calls these for .spr assets)
    def sprite_frames(self, asset: Asset):
        if asset.ref[0] != "sprframe" and not (asset.ref[0] == "bos" and asset.ext == "spr"):
            return [], 0.0
        data = self.raw(asset if asset.ref[0] == "bos" else asset.ref[1])
        try:
            info = spr_info(data)
        except Exception:
            return [], 0.0
        count = max(1, len(info["animations"]))
        step = 1.0 / 20.0  # .spr has no frame rate in the file; 20 fps reads well
        out = []
        for i in range(count):
            out.append((i * step, Asset("sprite", f"{asset.name} #{i}", ("sprframe", asset, i),
                                        uid=f"{asset.uid}#{i}", source=asset.source,
                                        ref=("sprframe", asset, i))))
        return out, count * step

    def animation_dim(self, asset):
        return "2d"  # every animation here is a flipbook

    def text(self, asset: Asset) -> str:
        if asset.ext == "ent":
            return self._ent_text(asset)
        return self.raw(asset).removeprefix(b"\xef\xbb\xbf").decode("utf-8", "replace")

    def _ent_text(self, asset: Asset) -> str:
        """Best-effort readable view of an .ent: the leading length-prefixed strings (the name and
        description) plus the raw hex of the rest, so the file is still explorable."""
        data = self.raw(asset)
        lines = []
        pos = 0
        for label in ("Name", "Description"):
            if pos + 4 > len(data):
                break
            length, = struct.unpack_from("<I", data, pos)
            if length > 512 or pos + 4 + length > len(data):
                break
            pos += 4
            value = data[pos:pos + length].decode("latin-1", "replace").rstrip("\0")
            pos += length
            lines.append(f"{label}: {value}")
        lines.append("")
        lines.append("Raw bytes:")
        for off in range(0, min(len(data), 512), 16):
            chunk = data[off:off + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            text_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{off:08x}  {hex_part:<47}  {text_part}")
        return "\n".join(lines)

    def audio(self, asset: Asset):
        if asset.ref[0] == "loose":
            return self.raw(asset), asset.ext
        raise NotImplementedError("sound inside a .bos archive: export the archive and play the file")

    def stats(self, asset: Asset) -> AssetStats:
        stats: AssetStats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        try:
            if asset.ext in ("zar", "til") and asset.ref[0] == "bos":
                data = self.raw(asset)
                at = 0 if asset.ext == "zar" else find_zar(data)
                if at >= 0:
                    w, h = zar_size(data, at)
                    stats.update(w=w, h=h, info=f"{w}\u00d7{h}", sort=w * h)
            elif asset.ext == "spr" and asset.ref[0] == "bos":
                info = spr_info(self.raw(asset))
                n = len(info["animations"])
                seqs = len(info["sequences"])
                stats.update(info=f"{n} anim, {seqs} seq", sort=n)
            elif asset.ext in LOOSE_IMAGES and asset.ref[0] == "loose":
                from PIL import Image
                w, h = Image.open(io.BytesIO(self.raw(asset))).size
                stats.update(w=w, h=h, info=f"{w}\u00d7{h}", sort=w * h)
        except Exception as e:
            log.debug("No stats for %s: %s", asset.name, e)
        return stats

    def describe(self, asset: Asset) -> list[tuple[str, str]]:
        rows = super().describe(asset)
        if asset.ref[0] == "bos":
            _t, bos_path, member = asset.ref
            rows.append(("Archive", os.path.basename(bos_path)))
            rows.append(("Stored as", member))
        elif asset.ref[0] == "sprframe":
            rows.append(("Frame", str(asset.ref[2])))
        elif asset.ref[0] == "loose":
            rows.append(("File", asset.ref[1]))
        if asset.ext == "spr" and asset.ref[0] == "bos":
            try:
                info = spr_info(self.raw(asset))
                rows.append(("Centre", f"{info['center'][0]}, {info['center'][1]}"))
                rows.append(("Sequences", ", ".join(s["name"] for s in info["sequences"][:8])))
                rows.append(("Animations", str(len(info["animations"]))))
            except Exception:
                pass
        return rows


# ---------------------------------------------------------------- plugin

class FalloutTacticsPlugin(EnginePlugin):
    id = "fallout_tactics"
    name = "Fallout Tactics: Brotherhood of Steel"
    version = "1.0"
    author = "UniView"
    description = ("Fallout Tactics (Micro Forté): .bos archives (ZIP containers), .zar / .til images, "
                   ".spr sprite animations, .ent entity definitions and the loose .bik movies.")

    @staticmethod
    def _root(path: str) -> str | None:
        """The game folder: the path itself, its parent, or the grandparent, so picking core/ or a
        subfolder of it still finds the game."""
        path = os.path.normpath(path)
        for p in (path, os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
            if os.path.isdir(p) and (
                    os.path.isdir(os.path.join(p, "core")) or
                    os.path.isfile(os.path.join(p, "bos.exe")) or
                    os.path.isfile(os.path.join(p, "Bos.exe"))):
                return p
        return None

    def detect(self, path: str) -> int:
        root = self._root(path)
        if root is None:
            return 0
        for exe in ("bos.exe", "Bos.exe", "fttools.exe", "FT Tools.exe"):
            if os.path.isfile(os.path.join(root, exe)):
                return 95
        core = os.path.join(root, "core")
        if os.path.isdir(core):
            try:
                if any(f.lower().endswith(".bos") for f in os.listdir(core)):
                    return 80
            except OSError:
                pass
        return 0

    def game_info(self, path: str) -> GameInfo:
        root = self._root(path) or path
        detail = "Fallout Tactics: Brotherhood of Steel"
        core = os.path.join(root, "core")
        if os.path.isdir(core):
            try:
                bos = [f for f in os.listdir(core) if f.lower().endswith(".bos")]
                detail += f" \u00b7 {len(bos)} archive(s)"
            except OSError:
                pass
        return {"engine_version": "Fallout engine (Micro Forté)", "detail": detail}

    def _bos_files(self, root: str) -> list[str]:
        """The .bos archives under core/, plus any loose .bos next to the game."""
        found = []
        for folder in (os.path.join(root, "core"), root):
            if not os.path.isdir(folder):
                continue
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                continue
            for n in names:
                if n.lower().endswith(".bos"):
                    full = os.path.join(folder, n)
                    if full not in found:
                        found.append(full)
        return found

    def count_files(self, path: str) -> int:
        root = self._root(path) or path
        return len(self._bos_files(root))

    def open(self, path: str, progress) -> FotSession:
        root = self._root(path)
        if root is None:
            raise FileNotFoundError(f"No Fallout Tactics data (a core/ folder or bos.exe) found in:\n{path}")
        session = FotSession(self, root)
        bos_files = self._bos_files(root)
        if not bos_files:
            raise FileNotFoundError(f"No .bos archives found under:\n{root}")
        progress.files([(os.path.basename(b), os.path.getsize(b), b) for b in bos_files])
        found: dict[str, int] = {}

        def add(asset: Asset) -> None:
            session.assets.append(asset)
            found[asset.kind] = found.get(asset.kind, 0) + 1

        for n, bos_path in enumerate(bos_files, 1):
            progress.file(n - 1)
            rel = os.path.relpath(bos_path, root).replace("\\", "/")
            progress(f"Reading {rel} ({n}/{len(bos_files)})", n - 1, len(bos_files))
            try:
                z = session._zip(bos_path)
            except (OSError, zipfile.BadZipFile) as e:
                log.warning("Archive %s could not be read: %s", rel, e)
                session.warnings.append(f"{rel}: {e}")
                continue
            log.info("Archive %s: %d files", rel, len(z.namelist()))
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                low = member.lower()
                ext = low.rsplit(".", 1)[-1] if "." in low else ""
                kind: Kind = KINDS.get(ext, "file")
                name = member.replace("\\", "/")
                add(Asset(kind=kind, name=name, key=("bos", bos_path, member),
                          uid=f"fot:{os.path.basename(bos_path)}:{name}",
                          size=z.getinfo(member).file_size, path=name,
                          source=os.path.basename(bos_path),
                          ref=("bos", bos_path, member), ext=ext or "bin"))
            progress.found(found)

        # Loose files next to the game (movies, config, screenshots, ...).
        progress("Looking for loose files ...", len(bos_files), len(bos_files) + 1)
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext in LOOSE_IMAGES:
                    kind = "texture"
                elif ext in LOOSE_TEXTS:
                    kind = "text"
                elif ext in ("bik",):
                    kind = "file"
                elif ext in ("wav", "ogg", "mp3"):
                    kind = "audio"
                else:
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root).replace("\\", "/")
                add(Asset(kind=kind, name=f"files/{rel}", key=("loose", rel.lower()),
                          uid=f"fot:loose:{rel.lower()}", size=os.path.getsize(full), path=rel,
                          source="loose file", ref=("loose", full), ext=ext))
        session.file_count = len(bos_files)
        if not session.assets:
            raise RuntimeError("Nothing could be listed from this game's archives.")
        return session


PLUGIN = FalloutTacticsPlugin()