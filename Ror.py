"""Risk of Rain (2013) plugin for UniView (GameMaker: Studio 1.x, Hopoo Games).

The game's data is one GameMaker "data.win" blob: an IFF-style file that is either a separate
data.win next to the exe or embedded inside the exe. It is not encrypted, only chunked:

  FORM <size>
    GEN8  general info (byte 1 = bytecode version)
    TXTR  list of texture pages, each an embedded PNG
    TPAG  texture page items: a rectangle on a page plus where it sits inside its sprite's box
    SPRT  sprites: name, size, origin and a list of TPAG pointers (one per frame)
    BGND  backgrounds: name + one TPAG pointer
    SOND  sounds: name + (best effort) the index of the sound's data in AUDO
    AUDO  audio blobs (WAV or OGG), each with a 4-byte length in front

All pointers inside the file are offsets from the start of the FORM, so an embedded blob works the
same as a standalone data.win once you know where the FORM starts.

What's shown: every texture page, every sprite (multi-frame sprites as GIF animations, with the
frames listed), backgrounds, and every embedded sound whose bytes are OGG or WAV. Loose
.ogg/.wav/.mp3 files next to the game (the music is usually loose) are listed too. Code, rooms and
objects are not read.

Written without the game files to test against, so the SPRT/TPAG layouts follow the GameMaker 1.x
format as documented by community tools and the sound-name lookup is a heuristic that falls back to
numbered names. If a field looks off, run with the console visible: every skipped item is logged.
"""
from __future__ import annotations

import io
import logging
import mmap
import os
import struct
import threading
from collections import OrderedDict

from engines.sdk import Asset, AssetStats, EnginePlugin, GameInfo, GameSession, pil_image_from_bytes

log = logging.getLogger("viewer.ror")

LOOSE_AUDIO = {"ogg", "wav", "mp3"}
LOOSE_IMAGES = {"png", "jpg", "jpeg", "bmp", "gif"}
PNG_SIG = b"\x89PNG\r\n\x1a\n"
FPS = 15  # GameMaker sets playback speed per object (image_speed), not per sprite; 15 fps reads well
PAGE_CACHE = 24  # texture pages kept in memory (they can be many MB each)
SOUND_CACHE = 8  # decoded audio blobs kept in memory (a few are several MB)


class _LRU(OrderedDict):
    """Tiny LRU: move to end on hit, drop the oldest on overflow. The old cache dropped the oldest entry on every
    insert, which threw away pages you had just used when scrolling up and down a long list."""

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


# ---------------------------------------------------------------- data.win

class DataWin:
    """A GameMaker data blob inside a memory-mapped file (standalone data.win or inside an exe)."""

    def __init__(self, path: str, base: int = 0) -> None:
        self.path = path
        self._f = open(path, "rb")
        self.buf = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.base = base
        if self.buf[base:base + 4] != b"FORM":
            raise ValueError("not a GameMaker data file (no FORM header)")
        self.form_size = struct.unpack_from("<I", self.buf, base + 4)[0]
        self.chunks: dict[str, tuple[int, int]] = {}  # id -> (offset of data from FORM start, size)
        pos, end = 8, 8 + self.form_size
        limit = len(self.buf) - base
        while pos + 8 <= min(end, limit):
            cid = self.buf[base + pos:base + pos + 4].decode("latin-1")
            size = struct.unpack_from("<I", self.buf, base + pos + 4)[0]
            self.chunks[cid] = (pos + 8, size)
            pos += 8 + size
        if "GEN8" not in self.chunks:
            raise ValueError("GameMaker data file has no GEN8 chunk")
        self._pointer_cache: dict[str, list[int]] = {}

    def close(self) -> None:
        try:
            self.buf.close()
        finally:
            self._f.close()

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.buf, self.base + off)[0]

    def i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.buf, self.base + off)[0]

    def string(self, ptr: int) -> str:
        start = self.base + ptr
        end = self.buf.find(b"\0", start)
        if end < 0:
            return ""
        return self.buf[start:end].decode("utf-8", "replace")

    def pointers(self, chunk: str) -> list[int]:
        """The pointer table at the start of a list chunk (count, then count offsets). Cached: the audio chunk's
        table used to be re-parsed for every single sound blob."""
        cached = self._pointer_cache.get(chunk)
        if cached is not None:
            return cached
        if chunk not in self.chunks:
            self._pointer_cache[chunk] = []
            return []
        off, size = self.chunks[chunk]
        count = self.u32(off)
        if count * 4 + 4 > size or count > 0x1000000:
            raise ValueError(f"{chunk}: pointer table larger than the chunk")
        out = list(struct.unpack_from(f"<{count}I", self.buf, self.base + off + 4))
        self._pointer_cache[chunk] = out
        return out

    @property
    def bytecode(self) -> int:
        off, _ = self.chunks["GEN8"]
        return self.buf[self.base + off + 1]

    # ---- textures
    def png(self, ptr: int) -> bytes:
        """The PNG starting at ptr, cut at its IEND chunk."""
        start = self.base + ptr
        if self.buf[start:start + 8] != PNG_SIG:
            raise ValueError("texture page is not a PNG")
        pos = start + 8
        while pos + 12 <= len(self.buf):
            length = struct.unpack_from(">I", self.buf, pos)[0]
            kind = self.buf[pos + 4:pos + 8]
            pos += 12 + length
            if kind == b"IEND":
                return bytes(self.buf[start:pos])
        raise ValueError("texture page has no IEND chunk")

    def tpag(self, ptr: int) -> tuple:
        """(x, y, w, h, target_x, target_y, target_w, target_h, bound_w, bound_h, page)."""
        if ptr + 22 > len(self.buf) - self.base:
            raise ValueError("TPAG pointer out of range")
        return struct.unpack_from("<11H", self.buf, self.base + ptr)

    # ---- audio
    def audio_blob(self, index: int) -> bytes:
        ptr = self.pointers("AUDO")[index]
        length = self.u32(ptr)
        start = self.base + ptr + 4
        if start + length > len(self.buf):
            raise ValueError("audio blob out of range")
        return bytes(self.buf[start:start + length])


def find_data(path: str) -> tuple[str, int] | None:
    """(file, offset of FORM) for a game folder, data.win or exe, or None. Walks one folder up when a file inside
    a subfolder was picked (so dropping Sprites/x.png still finds the game's data.win)."""
    start = os.path.normpath(path)
    seen: set[str] = set()
    while start not in seen:
        seen.add(start)
        found = _find_here(start)
        if found is not None:
            return found
        parent = os.path.dirname(start)
        if parent == start:
            break
        start = parent
    return None


def _find_here(path: str) -> tuple[str, int] | None:
    if os.path.isfile(path):
        found = _form_offset(path)
        return (path, found) if found is not None else None
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return None
    # data.win first (its FORM is usually at 0), then the executables.
    files = [os.path.join(path, n) for n in names if n.lower() == "data.win"]
    files += [os.path.join(path, n) for n in names
              if n.lower().endswith((".exe", ".win", ".unx")) and n.lower() != "data.win"]
    for f in files:
        found = _form_offset(f)
        if found is not None:
            return f, found
    return None


def _form_offset(path: str) -> int | None:
    try:
        size = os.path.getsize(path)
        if size < 64:
            return None
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            at = 0 if m[:4] == b"FORM" else m.find(b"FORM")
            tries = 0
            while at >= 0 and tries < 200:
                if at + 12 <= len(m) and m[at + 8:at + 12] == b"GEN8":
                    declared = struct.unpack_from("<I", m, at + 4)[0]
                    if at + 8 + declared <= size:
                        return at
                at = m.find(b"FORM", at + 4)
                tries += 1
    except (OSError, ValueError):
        pass
    return None


def gif_bytes(images: list, seconds: float) -> bytes:
    """An endless-loop animated GIF of RGBA PIL images (palette index 255 transparent where alpha < 50%)."""
    frames = []
    for img in images:
        mask = img.getchannel("A").point(lambda a: 255 if a < 128 else 0)
        p = img.convert("RGB").quantize(255)
        p.paste(255, mask=mask)
        frames.append(p)
    out = io.BytesIO()
    frames[0].save(out, "GIF", save_all=True, append_images=frames[1:], duration=max(20, round(seconds * 1000)),
                   loop=0, disposal=2, transparency=255, optimize=False)
    return out.getvalue()


# ---------------------------------------------------------------- session

class RorSession(GameSession):
    def __init__(self, plugin: EnginePlugin, path: str, data: DataWin) -> None:
        super().__init__(plugin, path)
        self.data = data
        self.root = path if os.path.isdir(path) else os.path.dirname(path)
        self._lock = threading.Lock()
        self._pages: _LRU = _LRU(PAGE_CACHE)
        self._sounds: _LRU = _LRU(SOUND_CACHE)
        self._page_ptrs: list[int] = []
        self._sprites: list[dict] = []
        self._by_key: dict | None = None

    def close(self) -> None:
        self.data.close()
        self._pages.clear()
        self._sounds.clear()

    # ---- images
    def page(self, index: int):
        from PIL import Image
        with self._lock:
            img = self._pages.get_lru(index)
            if img is None:
                img = Image.open(io.BytesIO(self.data.png(self._page_ptrs[index]))).convert("RGBA")
                self._pages.put(index, img)
            return img

    def tpag_image(self, ptr: int, canvas: tuple[int, int] | None = None):
        from PIL import Image
        x, y, w, h, tx, ty, tw, th, bw, bh, page = self.data.tpag(ptr)
        tile = self.page(page).crop((x, y, x + w, y + h))
        size = canvas or (max(bw, tx + tw, 1), max(bh, ty + th, 1))
        out = Image.new("RGBA", size, (0, 0, 0, 0))
        out.paste(tile, (tx, ty))
        return out

    def frame(self, sprite: int, index: int):
        s = self._sprites[sprite]
        if not s["frames"]:
            raise ValueError(f"sprite '{s['name']}' has no frames")
        index = max(0, min(index, len(s["frames"]) - 1))
        return self.tpag_image(s["frames"][index], (max(s["w"], 1), max(s["h"], 1)))

    def image(self, asset: Asset):
        tag = asset.ref[0]
        if tag == "page":
            return self.page(asset.ref[1]).copy()
        if tag == "spr":
            return self.frame(asset.ref[1], asset.ref[2])
        if tag == "frame":
            _tag, anim, index = asset.ref
            return self.frame(anim.ref[1], index)
        if tag == "anim":
            return self.frame(asset.ref[1], 0)
        if tag == "bg":
            return self.tpag_image(asset.ref[1])
        if tag == "loose":
            return pil_image_from_bytes(self.raw(asset))
        raise ValueError(f"not an image: {asset.name}")

    def sprite_frames(self, asset: Asset):
        if asset.ref[0] != "anim":
            return [], 0.0
        count = len(self._sprites[asset.ref[1]]["frames"])
        step = 1.0 / FPS
        out = []
        for i in range(count):
            out.append((i * step, Asset("sprite", f"{asset.name} #{i}", ("frame", asset.key, i),
                                        uid=f"{asset.uid}#{i}", source=asset.source, ref=("frame", asset, i))))
        return out, count * step

    def animation_dim(self, asset):
        return "2d"  # every animation here is a flipbook

    def _page_assets(self) -> dict:
        if self._by_key is None:
            self._by_key = {a.key: a for a in self.assets}
        return self._by_key

    def sprite_regions(self, asset: Asset):
        """Where a frame comes from on its texture page: (page asset, (x, y, w, h) on the page, 4 corners in the
        frame image, ordered top-left, top-right, bottom-right, bottom-left of the region)."""
        if asset.ref[0] != "frame" or asset.ref[1].ref[0] != "anim":
            return []
        _tag, anim, index = asset.ref
        sprite = self._sprites[anim.ref[1]]
        if not sprite["frames"]:
            return []
        index = max(0, min(index, len(sprite["frames"]) - 1))
        try:
            x, y, w, h, tx, ty, tw, th, _bw, _bh, page = self.data.tpag(sprite["frames"][index])
        except (ValueError, struct.error):
            return []
        sheet = self._page_assets().get(("page", page))
        if sheet is None:
            return []
        corners = [(float(tx), float(ty)), (float(tx + tw), float(ty)),
                   (float(tx + tw), float(ty + th)), (float(tx), float(ty + th))]
        return [(sheet, (float(x), float(y), float(w), float(h)), corners)]

    def text(self, asset: Asset):
        if asset.ref[0] == "anim":
            s = self._sprites[asset.ref[1]]
            n = len(s["frames"])
            pages = sorted({self.data.tpag(p)[10] for p in s["frames"]})
            lines = [asset.name, "",
                     f"{n} frames, {s['w']}\u00d7{s['h']} px, origin ({s['ox']}, {s['oy']}); shown at {FPS} fps "
                     f"({n / FPS:.2f} s).",
                     f"Sprite '{s['name']}' from {os.path.basename(self.data.path)}; frames are cut from texture "
                     f"page(s) {', '.join(str(p) for p in pages)}.",
                     "GameMaker stores no playback speed in the sprite itself (objects set image_speed), so the "
                     "speed here is a preview default.",
                     "Export saves it as an animated GIF."]
            return "\n".join(lines)
        return super().text(asset)

    def related(self, asset: Asset):
        tag = asset.ref[0]
        if tag == "frame":
            asset = asset.ref[1]
            tag = asset.ref[0]
        if tag not in ("anim", "spr"):
            return super().related(asset)
        s = self._sprites[asset.ref[1]]
        by_key = self._page_assets()
        sheets: list[Asset] = []
        for ptr in s["frames"]:
            try:
                page = self.data.tpag(ptr)[10]
            except (ValueError, struct.error):
                continue
            sheet = by_key.get(("page", page))
            if sheet is not None and sheet not in sheets:
                sheets.append(sheet)
        return "Sprite sheet", sheets, "No texture page found for this sprite."

    def animation_code(self, asset: Asset):
        if asset.ref[0] != "anim":
            return super().animation_code(asset)
        s = self._sprites[asset.ref[1]]
        lines = [f"{s['name']}: {len(s['frames'])} frames, {s['w']}\u00d7{s['h']} px, origin ({s['ox']}, {s['oy']})",
                 "GameMaker sprites are plain flipbooks: each frame is a rectangle of a texture page (TPAG entry)",
                 "drawn at an offset inside the sprite's box. No transforms or scripts are stored with the sprite.",
                 "", "frame  page  source rect (x, y, w, h)    placed at (x, y) size (w, h)"]
        for i, ptr in enumerate(s["frames"]):
            try:
                x, y, w, h, tx, ty, tw, th, _bw, _bh, page = self.data.tpag(ptr)
            except (ValueError, struct.error):
                lines.append(f"{i:>5}  (unreadable)")
                continue
            lines.append(f"{i:>5}  {page:>4}  ({x}, {y}, {w}, {h})".ljust(44) + f"({tx}, {ty}) ({tw}, {th})")
        return f"Animation code ({s['name']})", "\n".join(lines)

    # ---- audio / raw
    def _sound_bytes(self, index: int) -> bytes:
        with self._lock:
            data = self._sounds.get_lru(index)
            if data is None:
                data = self.data.audio_blob(index)
                self._sounds.put(index, data)
            return data

    def audio(self, asset: Asset):
        if asset.ref[0] == "loose":
            return self.raw(asset), asset.ext
        data = self._sound_bytes(asset.ref[1])
        if data[:4] == b"OggS":
            return data, "ogg"
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return data, "wav"
        raise NotImplementedError("unrecognized audio data (not OGG or WAV)")

    def raw(self, asset: Asset) -> bytes:
        tag = asset.ref[0]
        if tag == "loose":
            with open(asset.ref[1], "rb") as f:
                return f.read()
        if tag == "page":
            return self.data.png(self._page_ptrs[asset.ref[1]])
        if tag == "snd":
            return self._sound_bytes(asset.ref[1])
        if tag == "anim":
            count = len(self._sprites[asset.ref[1]]["frames"])
            return gif_bytes([self.frame(asset.ref[1], i) for i in range(count)], count / FPS)
        out = io.BytesIO()
        self.image(asset).save(out, "PNG")
        return out.getvalue()

    def stats(self, asset: Asset) -> AssetStats:
        stats: AssetStats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        try:
            tag = asset.ref[0]
            if tag == "page":
                w, h = self.page(asset.ref[1]).size
                stats.update(w=w, h=h, info=f"{w}\u00d7{h}", sort=w * h)
            elif tag in ("spr", "anim"):
                s = self._sprites[asset.ref[1]]
                n = len(s["frames"])
                stats.update(w=s["w"], h=s["h"], info=f"{s['w']}\u00d7{s['h']}" + (f", {n} frames" if n > 1 else ""),
                             sort=n if tag == "anim" else s["w"] * s["h"])
            elif tag == "frame":
                s = self._sprites[asset.ref[1].ref[1]]
                stats.update(w=s["w"], h=s["h"], info=f"{s['w']}\u00d7{s['h']}", sort=s["w"] * s["h"])
        except Exception as e:
            log.debug("No stats for %s: %s", asset.name, e)
        return stats

    def describe(self, asset: Asset) -> list[tuple[str, str]]:
        rows = super().describe(asset)
        tag = asset.ref[0]
        if tag == "frame":
            rows.append(("Frame", str(asset.ref[2])))
            tag, idx = "spr", asset.ref[1].ref[1]
        else:
            idx = asset.ref[1] if tag in ("spr", "anim") else None
        if idx is not None:
            s = self._sprites[idx]
            rows += [("Sprite", s["name"]), ("Origin", f"{s['ox']}, {s['oy']}"), ("Frames", str(len(s["frames"])))]
        if asset.ref[0] == "loose":
            rows.append(("File", asset.ref[1]))
        return rows


# ---------------------------------------------------------------- plugin

class RorPlugin(EnginePlugin):
    id = "ror"
    name = "Risk of Rain (2013)"
    version = "1.1"
    author = "Level7even Studios"
    description = ("Risk of Rain (2013, GameMaker: Studio 1.x): texture pages, sprites and animations, "
                   "backgrounds and sounds from data.win (or embedded in the exe). Not encrypted.")

    def detect(self, path: str) -> int:
        found = find_data(path)
        if found is None:
            return 0
        text = (path + " " + os.path.basename(found[0])).lower()
        return 95 if "risk of rain" in text or "riskofrain" in text or "ror" == os.path.splitext(
            os.path.basename(found[0]))[0].lower() else 60

    def game_info(self, path: str) -> GameInfo:
        found = find_data(path)
        detail = "Risk of Rain"
        if found:
            try:
                dw = DataWin(*found)
                detail += f" \u00b7 bytecode {dw.bytecode}"
                dw.close()
            except Exception:
                pass
        return {"engine_version": "GameMaker", "detail": detail}

    def count_files(self, path: str) -> int:
        return 1

    def open(self, path: str, progress) -> RorSession:
        found = find_data(path)
        if found is None:
            raise FileNotFoundError(f"No GameMaker data (data.win or an exe with embedded data) found in:\n{path}")
        file, offset = found
        data = DataWin(file, offset)
        session = RorSession(self, path, data)
        rel = os.path.basename(file)
        progress.files([(rel, os.path.getsize(file))])
        progress.file(0)
        progress(f"Reading {rel} ...", 0, 5)
        found_counts: dict[str, int] = {}

        def add(asset: Asset) -> None:
            session.assets.append(asset)
            found_counts[asset.kind] = found_counts.get(asset.kind, 0) + 1

        # texture pages
        session._page_ptrs = []
        for entry in data.pointers("TXTR"):
            session._page_ptrs.append(data.u32(entry + 4))  # (scaled, pointer to PNG)
        for i, ptr in enumerate(session._page_ptrs):
            try:
                size = len(data.png(ptr))
            except Exception as e:
                log.warning("Texture page %d unreadable: %s", i, e)
                continue
            add(Asset(kind="texture", name=f"texture pages/page_{i:03d}", key=("page", i), uid=f"ror:page:{i}",
                      size=size, path=f"texture pages/page_{i:03d}.png", source=rel, ref=("page", i), ext="png"))
        progress.found(found_counts)

        # sprites
        progress("Reading sprites ...", 1, 5)
        bad = 0
        for i, ptr in enumerate(data.pointers("SPRT")):
            try:
                name = data.string(data.u32(ptr))
                w, h = data.u32(ptr + 4), data.u32(ptr + 8)
                ox, oy = data.u32(ptr + 48), data.u32(ptr + 52)
                count = data.u32(ptr + 56)
                if count > 4096 or w > 16384 or h > 16384:
                    raise ValueError("implausible sprite header")
                frames = list(struct.unpack_from(f"<{count}I", data.buf, data.base + ptr + 60))
                if frames:
                    data.tpag(frames[0])  # validates the pointer
            except Exception as e:
                bad += 1
                log.debug("Sprite %d skipped: %s", i, e)
                session._sprites.append({"name": f"sprite_{i}", "w": 0, "h": 0, "ox": 0, "oy": 0, "frames": []})
                continue
            session._sprites.append({"name": name, "w": w, "h": h, "ox": ox, "oy": oy, "frames": frames})
            if not frames:
                continue
            multi = len(frames) > 1
            add(Asset(kind="animation" if multi else "sprite", name=f"sprites/{name}", key=("sprite", i),
                      uid=f"ror:sprite:{i}", size=None, path=f"sprites/{name}." + ("gif" if multi else "png"),
                      source=rel, ref=("anim", i) if multi else ("spr", i, 0), ext="gif" if multi else "png"))
        if bad:
            session.warnings.append(f"{bad} sprite(s) could not be read (see the console). If that is most of them, "
                                    "this build's sprite layout differs from the one this plugin expects.")
        progress.found(found_counts)

        # backgrounds
        progress("Reading backgrounds ...", 2, 5)
        for i, ptr in enumerate(data.pointers("BGND")):
            try:
                name = data.string(data.u32(ptr))
                tp = data.u32(ptr + 16)
                data.tpag(tp)
            except Exception as e:
                log.debug("Background %d skipped: %s", i, e)
                continue
            add(Asset(kind="texture", name=f"backgrounds/{name}", key=("bg", i), uid=f"ror:bg:{i}", size=None,
                      path=f"backgrounds/{name}.png", source=rel, ref=("bg", tp), ext="png"))

        # sounds
        progress("Reading sounds ...", 3, 5)
        audio_ptrs = data.pointers("AUDO")
        names = self._sound_names(data, len(audio_ptrs))
        undecodable = 0
        for i, ptr in enumerate(audio_ptrs):
            blob_len = data.u32(ptr)
            head = bytes(data.buf[data.base + ptr + 4:data.base + ptr + 16])
            if head[:4] == b"OggS":
                ext = "ogg"
            elif head[:4] == b"RIFF" and head[8:12] == b"WAVE":
                ext = "wav"
            else:
                # Not a format we can play; list it as a plain file so the UI doesn't offer a broken preview.
                undecodable += 1
                continue
            label = names.get(i) or f"audio_{i:04d}"
            add(Asset(kind="audio", name=f"sounds/{label}", key=("snd", i), uid=f"ror:snd:{i}", size=blob_len,
                      path=f"sounds/{label}.{ext}", source=rel, ref=("snd", i), ext=ext))
        if undecodable:
            log.info("Skipped %d audio blob(s) in an unrecognized format (not OGG or WAV)", undecodable)
        progress.found(found_counts)

        # loose files next to the game (music is usually loose .ogg)
        progress("Looking for loose files ...", 4, 5)
        root = session.root
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext in LOOSE_AUDIO:
                    kind = "audio"
                elif ext in LOOSE_IMAGES:
                    kind = "texture"
                else:
                    continue
                full = os.path.join(dirpath, fname)
                relpath = os.path.relpath(full, root).replace("\\", "/")
                add(Asset(kind=kind, name=f"files/{relpath}", key=("loose", relpath.lower()),
                          uid=f"ror:loose:{relpath.lower()}", size=os.path.getsize(full), path=relpath,
                          source="loose file", ref=("loose", full), ext=ext))
        session.file_count = 1
        if not session.assets:
            raise RuntimeError("Nothing could be listed from this game's data file.")
        return session

    @staticmethod
    def _sound_names(data: DataWin, audio_count: int) -> dict[int, str]:
        """audio index -> sound name. The SOND layout varies between GameMaker 1.x builds, so find the
        field that holds the audio index by checking which candidate offset gives mostly-unique, in-range
        values; if none does, return nothing (sounds keep numbered names)."""
        try:
            ptrs = data.pointers("SOND")
        except Exception:
            return {}
        if not ptrs or not audio_count:
            return {}
        best: tuple[int, int, float] | None = None
        for off in (36, 32, 28, 40, 24, 44, 48):
            try:
                ids = [data.i32(p + off) for p in ptrs]
            except struct.error:
                continue
            valid = [i for i in ids if 0 <= i < audio_count]
            if len(valid) < max(1, len(ptrs) // 2):
                continue
            score = len(set(valid)) / len(valid)
            if score >= 0.9 and (best is None or len(valid) > best[0]):
                best = (len(valid), off, score)
        if best is None:
            log.info("Sound names: no audio-index field found in SOND, using numbered names")
            return {}
        log.info("Sound names: using SOND offset +%d (%d/%d names, %.2f unique)",
                 best[1], best[0], len(ptrs), best[2])
        out: dict[int, str] = {}
        for p in ptrs:
            try:
                idx = data.i32(p + best[1])
                if 0 <= idx < audio_count:
                    name = data.string(data.u32(p))
                    if name:
                        out.setdefault(idx, name)
            except Exception:
                continue
        return out


PLUGIN = RorPlugin()