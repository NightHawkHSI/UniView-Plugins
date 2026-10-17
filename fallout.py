"""Fallout 1 & 2 (Interplay's engine): .dat archives, FRM sprites, RIX images, ACM sounds, MSG text.

DAT2 (Fallout 2) stores its directory at the end and compresses with zlib; DAT1 (Fallout 1) is
big-endian with LZSS compression. Sprites use the game's 256-colour palette (color.pal).
"""

import logging
import os
import struct
import threading
import zlib

import numpy as np
from PIL import Image

from engines.sdk import Asset, EnginePlugin, GameSession, Kind, listdir_lower

log = logging.getLogger("viewer.fallout")

KINDS: dict[str, Kind] = {
    "frm": "sprite", "fr0": "sprite", "fr1": "sprite", "fr2": "sprite", "fr3": "sprite", "fr4": "sprite",
    "fr5": "sprite", "rix": "texture", "acm": "audio", "msg": "text", "txt": "text", "lst": "text",
    "gam": "text", "cfg": "text", "ini": "text", "sve": "text", "bio": "text", "gcd": "file"}


def lzss_decompress(data, size):
    """Fallout 1's LZSS: blocks of [int16 BE n]; n < 0 -> -n raw bytes, n > 0 -> n bytes of LZSS."""
    out = bytearray()
    pos = 0
    while pos + 2 <= len(data) and len(out) < size:
        n, = struct.unpack_from(">h", data, pos)
        pos += 2
        if n == 0:
            break
        if n < 0:
            out += data[pos:pos - n]
            pos -= n
            continue
        end = pos + n
        window = bytearray(b" " * 4096)
        wpos = 4078
        while pos < end:
            flags = data[pos]
            pos += 1
            for bit in range(8):
                if pos >= end:
                    break
                if flags & (1 << bit):
                    b = data[pos]
                    pos += 1
                    out.append(b)
                    window[wpos] = b
                    wpos = (wpos + 1) & 4095
                else:
                    if pos + 1 >= end + 1:
                        break
                    lo, hi = data[pos], data[pos + 1]
                    pos += 2
                    offset = lo | ((hi & 0xF0) << 4)
                    length = (hi & 0x0F) + 3
                    for k in range(length):
                        b = window[(offset + k) & 4095]
                        out.append(b)
                        window[wpos] = b
                        wpos = (wpos + 1) & 4095
    return bytes(out[:size])


class DatEntry:
    __slots__ = ("name", "compressed", "size", "packed", "offset")

    def __init__(self, name, compressed, size, packed, offset):
        self.name, self.compressed, self.size, self.packed, self.offset = name, compressed, size, packed, offset


class Dat:
    """A Fallout 1 or 2 .dat archive. read() is thread-safe."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        self._lock = threading.Lock()
        self.entries = []
        f = self._f
        f.seek(0, 2)
        total = f.tell()
        f.seek(total - 8)
        tree_size, data_size = struct.unpack("<II", f.read(8))
        if data_size == total and 0 < tree_size < total:
            self.version = 2
            f.seek(total - tree_size - 8)
            tree = f.read(tree_size)
            count, = struct.unpack_from("<I", tree, 0)
            pos = 4
            for _ in range(count):
                n, = struct.unpack_from("<I", tree, pos)
                name = tree[pos + 4:pos + 4 + n].decode("latin-1").replace("\\", "/")
                pos += 4 + n
                comp, size, packed, offset = struct.unpack_from("<BIII", tree, pos)
                pos += 13
                self.entries.append(DatEntry(name, comp == 1, size, packed, offset))
            return
        self.version = 1
        f.seek(0)
        head = f.read(16)
        dir_count, = struct.unpack(">I", head[:4])
        if not 0 < dir_count < 10000:
            raise ValueError("Not a Fallout .dat file")
        buf = f.read(min(total, 8 * 1024 * 1024))
        pos = 0
        dirs = []
        for _ in range(dir_count):
            n = buf[pos]
            dirs.append(buf[pos + 1:pos + 1 + n].decode("latin-1").replace("\\", "/"))
            pos += 1 + n
        for d in dirs:
            file_count, = struct.unpack_from(">I", buf, pos)
            pos += 16
            for _ in range(file_count):
                n = buf[pos]
                name = buf[pos + 1:pos + 1 + n].decode("latin-1")
                pos += 1 + n
                attr, offset, size, packed = struct.unpack_from(">IIII", buf, pos)
                pos += 16
                full = name if d in (".", "") else f"{d}/{name}"
                self.entries.append(DatEntry(full, attr == 0x40, size, packed, offset))

    def read(self, entry):
        with self._lock:
            self._f.seek(entry.offset)
            raw = self._f.read(entry.packed if entry.compressed else entry.size)
        if not entry.compressed:
            return raw
        if self.version == 2:
            return zlib.decompress(raw)
        return lzss_decompress(raw, entry.size)

    def close(self):
        with self._lock:
            self._f.close()


# --------------------------------------------------------------------------- images

def palette_from(data):
    pal = np.frombuffer(data[:768], np.uint8).reshape(256, 3).astype(np.int32)
    pal = np.clip(pal * 4, 0, 255).astype(np.uint8)
    rgba = np.concatenate([pal, np.full((256, 1), 255, np.uint8)], axis=1)
    rgba[0, 3] = 0  # index 0 is transparent
    return rgba


def frm_frames(data):
    """[(direction, [(w, h, pixel indices)])] of an FRM."""
    fpd, = struct.unpack_from(">H", data, 8)
    offsets = struct.unpack_from(">6I", data, 34)
    base = 62
    out = []
    seen = set()
    for d, off in enumerate(offsets):
        if off in seen or (d and off == 0):
            continue
        seen.add(off)
        pos = base + off
        frames = []
        for _ in range(max(fpd, 1)):
            if pos + 12 > len(data):
                break
            w, h, size = struct.unpack_from(">HHI", data, pos)
            pos += 12
            frames.append((w, h, np.frombuffer(data, np.uint8, w * h, pos)))
            pos += size
        out.append((d, frames))
    return out


def frm_image(data, palette):
    """First direction's frames side by side (the whole animation at a glance)."""
    dirs = frm_frames(data)
    if not dirs or not dirs[0][1]:
        raise ValueError("Empty FRM")
    frames = dirs[0][1][:24]
    width = sum(w for w, _h, _p in frames) + 2 * (len(frames) - 1)
    height = max(h for _w, h, _p in frames)
    sheet = Image.new("RGBA", (max(width, 1), max(height, 1)), (0, 0, 0, 0))
    x = 0
    for w, h, px in frames:
        if w and h:
            sheet.paste(Image.fromarray(palette[px.reshape(h, w)], "RGBA"), (x, height - h))
        x += w + 2
    return sheet


def rix_image(data):
    if data[:4] != b"RIX3":
        raise ValueError("Not a RIX image")
    w, h = struct.unpack_from("<HH", data, 4)
    pal = palette_from(data[10:10 + 768])
    pal[0, 3] = 255
    px = np.frombuffer(data, np.uint8, w * h, 10 + 768)
    return Image.fromarray(pal[px.reshape(h, w)], "RGBA")


# --------------------------------------------------------------------------- session

class FalloutSession(GameSession):
    def __init__(self, plugin, path):
        super().__init__(plugin, path)
        self.dats = []
        self.files = {}   # lowercase path -> (Dat or None, entry or disk path)
        self._palette = None

    def close(self):
        for d in self.dats:
            d.close()

    def _read(self, key):
        dat, entry = self.files[key]
        if dat is None:
            with open(entry, "rb") as f:
                return f.read()
        return dat.read(entry)

    def raw(self, asset):
        return self._read(asset.ref)

    def palette(self):
        if self._palette is None:
            key = "color.pal"
            self._palette = palette_from(self._read(key)) if key in self.files else \
                palette_from(bytes(range(256)) * 3)
        return self._palette

    def image(self, asset):
        data = self.raw(asset)
        if asset.ext == "rix":
            return rix_image(data)
        return frm_image(data, self.palette())

    def audio(self, asset):
        from engines import extdecode
        return extdecode.to_wav(self.raw(asset), "acm"), "wav"

    def text(self, asset):
        return self.raw(asset).decode("latin-1")

    def stats(self, asset):
        stats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        if asset.kind == "sprite":
            data = self.raw(asset)
            dirs = frm_frames(data)
            if dirs and dirs[0][1]:
                w, h, _p = dirs[0][1][0]
                n = len(dirs[0][1])
                stats.update(w=w, h=h, info=f"{w}×{h}" + (f", {n} frames" if n > 1 else ""), sort=w * h)
        return stats

    def describe(self, asset):
        dat, _entry = self.files[asset.ref]
        rows = [("File", os.path.basename(dat.path) if dat else "loose file"), ("Path", asset.path)]
        if asset.kind == "sprite":
            try:
                dirs = frm_frames(self.raw(asset))
                rows.append(("Animation", f"{len(dirs)} direction(s), {len(dirs[0][1])} frame(s) each "
                                          "(the first direction is shown)"))
            except Exception:
                pass
        return rows


class FalloutPlugin(EnginePlugin):
    id = "fallout"
    name = "Fallout 1/2"
    version = "1.0"
    author = "UniView"
    description = "Fallout 1 and 2: .dat archives, FRM sprites, RIX images, ACM sounds (via vgmstream), MSG text."

    @staticmethod
    def _dats(path):
        names = listdir_lower(path)
        found = [os.path.join(path, names[n]) for n in ("master.dat", "critter.dat", "patch000.dat", "f2_res.dat")
                 if n in names]
        return found

    def detect(self, path):
        if os.path.isfile(path):
            return 80 if os.path.basename(path).lower() in ("master.dat", "critter.dat") else 0
        return 95 if {"master.dat", "critter.dat"} <= set(listdir_lower(path)) else 0

    def game_info(self, path):
        try:
            dat = Dat(self._dats(path)[0])
            version = f"DAT{dat.version}"
            dat.close()
        except Exception:
            version = ""
        return {"engine_version": "", "detail": version}

    def label(self, info):
        return "Fallout 1/2"

    def count_files(self, path):
        return len(self._dats(path))

    def open(self, path, progress):
        session = FalloutSession(self, path)
        root = path if os.path.isdir(path) else os.path.dirname(path)
        dats = [path] if os.path.isfile(path) else self._dats(path)
        # Loose files in data/ override the archives (that's how patches work).
        data_dir = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else \
            os.path.join(root, "DATA")
        if os.path.isdir(data_dir):
            for dirpath, _dirs, names in os.walk(data_dir):
                for fname in names:
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, data_dir).replace("\\", "/").lower()
                    if not rel.endswith(".dat"):
                        session.files.setdefault(rel, (None, full))
        progress.files((os.path.basename(d), os.path.getsize(d) if os.path.isfile(d) else 0, d) for d in dats)
        for n, dat_path in enumerate(dats, 1):
            progress.file(n - 1)
            progress(f"Reading {os.path.basename(dat_path)} ({n}/{len(dats)})", n - 1, len(dats))
            try:
                dat = Dat(dat_path)
            except Exception as e:
                log.warning("Could not read %s: %s", dat_path, e)
                continue
            session.dats.append(dat)
            for entry in dat.entries:
                session.files.setdefault(entry.name.lower(), (dat, entry))
        progress.file(len(dats))
        session.file_count = len(session.dats)
        for key, (dat, entry) in session.files.items():
            ext = key.rsplit(".", 1)[-1] if "." in key.rsplit("/", 1)[-1] else ""
            kind = KINDS.get(ext, "file")
            size = entry.size if dat is not None else os.path.getsize(entry)
            name = key.rsplit(".", 1)[0] if kind in ("sprite", "texture") else key
            session.assets.append(Asset(kind, name, key, uid=key, size=size, path=key,
                                        source=os.path.basename(dat.path) if dat else "", ref=key, ext=ext))
        if not session.assets:
            raise FileNotFoundError(f"No Fallout .dat files found in:\n{path}")
        return session


PLUGIN = FalloutPlugin()
