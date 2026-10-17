"""Wakfu engine plugin for UniView (Ankama's Java client).

Everything lives under contents/, mostly in zip files named .jar:
  - animations/<group>/<group>.jar: Ankama's 2D animations (.anm) plus their texture atlases (Atlas/*.png)
  - gui_jar/gui.jar, particles/particles.jar, hm/hm.jar, shaders/shaders.jar: TGA / PNG / DDS images, fonts, shaders
  - maps/gfx.jar: the map element graphics (.tgam: a small header + raw RGBA at the end)
  - sounds/<group>/<group>.pk: sound packs (Ogg Vorbis files with every byte +1)
  - scripts/scripts.jar (Lua), i18n/*.jar (texts), data/data.jar (XML), videos/videos.jar (MKV)

.anm ("Anm2") layout, worked out from the files themselves: header, an index of animation names (and of other .anm files holding more of them), the atlas name, shapes (atlas
rectangles), transform tables (colors, 2x2 matrices, translations), actions (sounds, jumps), sprite definitions and
imports (parts taken from another .anm). A sprite definition lists, per frame, child ids (shapes, other sprites or
imports) and a byte/short/int stream: per child an opcode whose bits say which tables it reads - 1 matrix,
2 translation, 4 color multiply, 8 color add. Named sprites ("5_AnimStatique": direction 5 of AnimStatique) are the
animations. Player classes keep their parts in one .anm (10.anm ...) and most animations in shared files that import
those parts by name; such files are drawn with the parts of the first class .anm that lists them.

What's shown: animations (flipbooks, exported as GIF), a still of every .anm (equipment pieces and other part
libraries as a sheet of their parts), all images (including DDS), sounds, fonts, videos and text files. NPC and pet
animations are listed by their gfx id (the .anm file name). Maps and the game data (bdata) are not read."""
from __future__ import annotations

import io
import os
import re
import struct
import threading
import zipfile
from collections import OrderedDict
from typing import Any

import numpy as np

from engines.sdk import Asset, AssetStats, EnginePlugin, GameInfo, GameSession, Kind, kind_for_extension, \
    pil_image_from_bytes

TEXT_EXTS = {"xml", "lua", "properties", "glsl", "cgfx", "txt"}
IMAGE_EXTS = {"png", "tga", "dds", "jpg"}
SKIP_DIRS = {"bdata", "maps"}  # maps/gfx.jar is read on its own; map chunks + game data aren't decoded yet
SKIP_PREFIXES = ("META-INF/",)  # every jar ships these
SKIP_SUFFIXES = (".class", ".jar", ".bat", ".sh")  # not assets
DIRECTION_ORDER = "13204756"  # which direction to show first: facing the camera (1 SE, 3 SW, 2 S) before the back
_DIR_NAME = re.compile(r"^(\d)_(.+)$")
MIN_SCALE, MAX_SCALE = 2.0, 4.0  # drawing scale range (shape units -> pixels)
ATLAS_CACHE, CLIP_CACHE = 64, 32  # caches kept per session (biggest-first, least-recently-used evicted)
SHEET_TILE_LIMIT = 96              # part tiles shown per .anm still


# ---------------------------------------------------------------- sound packs (.pk)

def pk_entries(f) -> list[tuple[int, int, int]]:
    """[(id, data offset, size)] of a .pk sound pack: u32 table size, u32 count, (u32 id, u32 offset) each, then
    per entry a u64 size and the Ogg file with every byte +1. Entries whose table offset runs past the file are
    dropped (some builds pad the table)."""
    head = f.read(8)
    if len(head) < 8:
        return []
    _size, count = struct.unpack("<II", head)
    try:
        end = f.seek(0, os.SEEK_END)
        f.seek(8)
    except OSError:
        return []
    table = f.read(count * 8)
    base = 8 + count * 8
    out = []
    for i in range(count):
        if (i + 1) * 8 > len(table):
            break
        sid, off = struct.unpack_from("<II", table, i * 8)
        entry_at = base + off
        if entry_at + 8 > end:
            continue
        f.seek(entry_at)
        size = struct.unpack("<Q", f.read(8))[0]
        if entry_at + 8 + size > end:
            continue
        out.append((sid, entry_at + 8, size))
    return out


_UNSHIFT = bytes((i - 1) & 255 for i in range(256))


def pk_ogg(data: bytes) -> bytes:
    """Ogg bytes of a .pk entry's data (every byte in the pack is shifted +1)."""
    return data.translate(_UNSHIFT)


# ---------------------------------------------------------------- map graphics (.tgam)

def tgam_image(data: bytes):
    """PIL image of a .tgam: "mAGT", u16 width, u16 height, u32 pixel bytes, u32 mask bytes, ..., then RGBA rows
    of a power-of-two texture at the end of the file (the element sits in its top-left corner)."""
    from PIL import Image
    if data[:4] != b"mAGT":
        raise ValueError("not a TGAM file")
    w, h, size, _mask = struct.unpack_from("<HHII", data, 4)
    tw = 1 << max(0, (w - 1).bit_length())
    th = size // 4 // tw if tw else 0
    if tw * th * 4 != size or size > len(data) or th < h:
        raise ValueError(f"unexpected TGAM layout ({w}x{h}, {size} pixel bytes)")
    img = Image.frombytes("RGBA", (tw, th), data[len(data) - size:])
    return img.crop((0, 0, w, h))


def tgam_size(head: bytes) -> tuple[int, int]:
    return struct.unpack_from("<HH", head, 4)


# ---------------------------------------------------------------- DDS images

_DDS_BCN = {b"DXT1": 1, b"DXT3": 2, b"DXT5": 3}
# DXGI formats (from the DX10 extended header) -> BCN mode
_DXGI_BCN = {70: 1, 71: 1, 72: 1, 73: 2, 74: 2, 75: 2, 76: 3, 77: 3, 78: 3,
             79: 3, 80: 3, 81: 3, 82: 3, 83: 3, 84: 3}
_DXGI_RGBA = {28: "RGBA", 29: "RGBA", 87: "BGRA", 88: "BGRA"}


def dds_image(data: bytes):
    """PIL image of a DDS file (DXT1/3/5 and uncompressed 24/32-bit, legacy and DX10 headers). Pillow doesn't ship
    a DDS plugin by default, so this decodes the common cases Ankama's tools produce."""
    from PIL import Image
    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    height, width = struct.unpack_from("<II", data, 12)
    if not width or not height:
        raise ValueError("empty DDS image")
    fourcc = data[84:88]
    bit_count = struct.unpack_from("<I", data, 88)[0]
    if fourcc in _DDS_BCN:
        return Image.frombytes("RGBA", (width, height), data[128:], "bcn", _DDS_BCN[fourcc]).crop((0, 0, width, height))
    if fourcc == b"DX10":
        if len(data) < 148:
            raise ValueError("truncated DX10 header")
        dxgi, _res, _arr, _misc2 = struct.unpack_from("<4I", data, 128)
        bcn = _DXGI_BCN.get(dxgi)
        if bcn:
            return Image.frombytes("RGBA", (width, height), data[148:], "bcn", bcn).crop((0, 0, width, height))
        raw = _DXGI_RGBA.get(dxgi)
        if raw:
            return Image.frombytes("RGBA", (width, height), data[148:], "raw", raw)
        raise NotImplementedError(f"DDS DXGI format {dxgi} isn't supported yet")
    if fourcc == b"\0\0\0\0" and bit_count in (24, 32):
        r_mask, g_mask, b_mask, a_mask = struct.unpack_from("<4I", data, 92)
        # Standard D3D layouts: A8R8G8B8 (BGRA in memory) or A8B8G8R8 (RGBA in memory).
        raw = "BGRA" if r_mask == 0x00FF0000 and b_mask == 0x000000FF else "RGBA"
        if bit_count == 24:
            return Image.frombytes("RGB", (width, height), data[128:], "raw", raw[:3])
        img = Image.frombytes("RGBA", (width, height), data[128:], "raw", raw)
        return img.convert("RGB") if a_mask == 0 else img
    raise NotImplementedError(f"DDS format {fourcc!r} / {bit_count} bits isn't supported yet")


# ---------------------------------------------------------------- animations (.anm)

class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, fmt: str) -> Any:
        v = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def array(self, code: str, n: int) -> list:
        if n <= 0:
            return []
        v = struct.unpack_from(f"<{n}{code}", self.data, self.pos)
        self.pos += struct.calcsize(code) * n
        return list(v)

    def string(self) -> str:
        end = self.data.index(b"\0", self.pos)
        s = self.data[self.pos:end].decode("utf-8", "replace")
        self.pos = end + 1
        return s


def _read_index(r: _Reader) -> tuple[int, list[str], list[tuple[str, int]], float]:
    """(header flags, other .anm files, [(animation name, file index or -1)], scale) - the start of an .anm."""
    flags = r.take("B")
    r.take("h")
    fps = r.take("B")
    f = r.take("B")
    scale = r.take("f") if f & 1 else 1.0
    if f & 8:
        r.take("f")  # radius
    files = [r.string() for _ in range(r.take("H"))] if f & 2 else []
    if f & 4:
        r.array("i", r.take("B") * 2)  # hidden parts
    if f & 0x40:
        for _ in range(r.take("B")):  # parts an equipment can hide
            r.string()
            r.take("i")
    if f & 0x80:
        mf = r.take("i")
        if mf & 1:
            for _ in range(r.take("H")):
                r.take("iB")
        if mf & 2:
            r.array("f", 3)
    anims = []
    for _ in range(r.take("H")):
        name = r.string()
        r.take("i")
        anims.append((name, r.take("h")))
    return flags | fps << 8, files, anims, scale


def anm_index(data: bytes) -> tuple[list[str], list[tuple[str, int]]]:
    """(other .anm files, [(animation name, file index)]) from the start of an .anm; raises if `data` is too short."""
    _flags, files, anims, _scale = _read_index(_Reader(data))
    return files, anims


class Anm:
    """A parsed .anm. Shapes: {id: (u0, v0, u1, v1, w, h, offset x, offset y)} (atlas fractions, v from the top);
    sprites: {id: dict}; imports: {id: part name}."""

    def __init__(self, data: bytes) -> None:
        r = _Reader(data)
        flags, self.files, self.anims, self.scale = _read_index(r)
        self.fps = (flags >> 8) or 25
        self.texture = ""
        if r.take("h") == 1:
            self.texture = r.string()
            r.take("i")
        self.shapes: dict[int, tuple] = {}
        for _ in range(r.take("H")):
            sid, _tex, top, left, bottom, right, w, h, ox, oy = r.take("hhHHHHhhff")
            self.shapes[sid] = (left / 65535, top / 65535, right / 65535, bottom / 65535, w, h, ox, oy)
        self.colors = r.array("f", r.take("i"))
        self.matrices = r.array("f", r.take("i"))
        self.translations = r.array("f", r.take("i"))
        for _ in range(r.take("i")):
            self._skip_action(r)
        self.sprites: dict[int, dict] = {}
        self.named: dict[str, int] = {}
        for _ in range(r.take("H")):
            sp = self._read_sprite(r)
            self.sprites[sp["id"]] = sp
            if sp["name"]:
                self.named.setdefault(sp["name"], sp["id"])
        self.imports: dict[int, str] = {}
        for _ in range(r.take("H")):
            iid = r.take("h")
            self.imports[iid] = r.string()
            r.take("i")
        self._parts: dict[str, int] | None = None

    @staticmethod
    def _skip_action(r: _Reader) -> None:
        kind, n = r.take("BB")
        if kind == 1:  # go to animation (name, percentage)
            r.string()
            if n == 2:
                r.take("B")
        elif kind == 3:  # sound id
            r.string()
        elif kind == 4 and n:  # random animation
            if r.string() == "#optimized":
                k = (n - 1) // 2
                for _ in range(k):
                    r.string()
                r.array("B", k)
            else:
                for _ in range(n - 1):
                    r.string()
        elif kind == 8:  # name pairs
            for _ in range((n - 1) // 2 * 2 + n % 2):
                r.string()
        elif kind == 9:
            r.take("i")
            r.array("h", 2 if n == 3 else 3 if n == 4 else 0)
        elif kind == 10:
            r.take("B")

    @staticmethod
    def _read_sprite(r: _Reader) -> dict:
        kind = r.take("B")
        sid, flags = r.take("hB")
        sp: dict[str, Any] = {"kind": kind, "id": sid, "flags": flags, "name": r.string() if flags & 64 else ""}
        r.take("ii")
        if kind == 1:  # one child (with actions)
            sp["children"] = [r.take("h")]
            r.array("h", r.take("H"))
        elif kind == 2:  # one child
            sp["children"] = [r.take("h")]
        elif kind == 3:  # one frame, several children
            sp["children"] = r.array("h", r.take("H"))
            r.array("h", r.take("H"))
        elif kind == 4:  # several frames
            sp["frames"] = r.array("i", r.take("H"))
            sp["childlist"] = r.array("h", r.take("H"))
            sp["stride"] = 3 if r.array("h", r.take("H")) else 2
        else:
            raise ValueError(f"unknown sprite kind {kind}")
        code, n = r.take("Bi")
        sp["data"] = r.array({1: "B", 2: "H", 4: "I"}.get(code, "B"), n)
        return sp

    # ---- playing
    def frame_count(self, sid: int) -> int:
        """Frames of a sprite; one with a single child (kinds 1 and 2) plays as long as that child."""
        sp = self.sprites.get(sid)
        for _ in range(16):
            if sp is None or sp["kind"] not in (1, 2):
                break
            sp = self.sprites.get(sp["children"][0])
        if sp is None or sp["kind"] != 4:
            return 1
        return max(1, len(sp["frames"]) // sp["stride"])

    def part(self, name: str) -> int | None:
        """Sprite id of a part another .anm imports (by full name, else by the name without its "N_" prefix)."""
        if self._parts is None:
            self._parts = {}
            for sp in self.sprites.values():
                if sp["name"]:
                    self._parts.setdefault(sp["name"], sp["id"])
                    self._parts.setdefault(sp["name"].split("_", 1)[-1], sp["id"])
        found = self._parts.get(name)
        return found if found is not None else self._parts.get(name.split("_", 1)[-1])

    def collect(self, sid: int, frame: int, xf: tuple, out: list, host: Anm | None = None, depth: int = 0) -> None:
        """Append (anm, shape, transform) for everything sprite `sid` draws at `frame`. transform: (a, b, c, d, tx, ty,
        rgba multiplier); a point (x, y) of a shape lands at (a x + c y + tx, b x + d y + ty), y down."""
        if depth > 40:
            return
        if sid in self.imports:  # a part of the main .anm: the host, else this file itself
            main = host if host is not None else self
            hid = main.part(self.imports[sid])
            if hid is not None and hid not in main.imports:
                main.collect(hid, frame, xf, out, None, depth + 1)
            return
        sp = self.sprites.get(sid)
        if sp is None:
            shape = self.shapes.get(sid)
            if shape is not None:
                out.append((self, shape, xf))
            return
        data = sp["data"]
        if sp["kind"] == 4:
            n = self.frame_count(sid)
            # Past its end a sprite holds its last frame, unless flag 128 (loop) says to start over.
            i = (frame % n if sp["flags"] & 128 else min(frame, n - 1)) * sp["stride"]
            if i + 1 >= len(sp["frames"]):
                return
            pos, at = sp["frames"][i], sp["frames"][i + 1]
            if at + 1 > len(sp["childlist"]):
                return
            kids = sp["childlist"][at + 1:at + 1 + sp["childlist"][at]]
        else:
            pos, kids = 0, sp.get("children", [])
        for kid in kids:
            if pos >= len(data):
                break
            op = data[pos]
            try:
                child, pos = self._apply(op, data, pos + 1, xf)
            except (IndexError, ValueError):
                break  # a truncated stream: draw what was read
            self.collect(kid, frame, child, out, host, depth + 1)

    def _apply(self, op: int, data: list, pos: int, parent: tuple) -> tuple[tuple, int]:
        a, b, c, d, tx, ty, col = parent
        tab = self.colors
        if op & 12 == 12:
            add, mul = data[pos], data[pos + 1]
            pos += 2
            col = tuple(col[k] * tab[mul + k] + tab[add + k] for k in range(4))
        elif op & 4:
            mul = data[pos]
            pos += 1
            col = tuple(col[k] * tab[mul + k] for k in range(4))
        elif op & 8:
            add = data[pos]
            pos += 1
            col = tuple(col[k] + tab[add + k] for k in range(4))
        na, nb, nc, nd, ntx, nty = a, b, c, d, tx, ty
        if op & 1:
            m = data[pos]
            pos += 1
            f2, f3, f4, f5 = self.matrices[m:m + 4]
            na, nb = f2 * a + f3 * c, f2 * b + f3 * d
            nc, nd = f4 * a + f5 * c, f4 * b + f5 * d
        if op & 2:
            t = data[pos]
            pos += 1
            x, y = self.translations[t:t + 2]
            ntx, nty = x * a + y * c + tx, x * b + y * d + ty
        return (na, nb, nc, nd, ntx, nty, col), pos

    # ---- the animation as text
    def code(self, sid: int, max_lines: int = 5000) -> list[str]:
        """Sprite `sid` and the sprites under it as text: per frame, each child and the transform its opcode gives
        it. This is what gets drawn every frame."""
        lines: list[str] = []
        todo, seen = [sid], {sid}
        while todo and len(lines) < max_lines:
            sp = self.sprites.get(todo.pop(0))
            if sp is None:
                continue
            frames = max(1, len(sp["frames"]) // sp["stride"]) if sp["kind"] == 4 else 1
            lines.append(f"sprite {sp['id']}" + (f' "{sp["name"]}"' if sp["name"] else "")
                         + f": kind {sp['kind']}, {frames} frame{'s' * (frames != 1)}"
                         + ((", loops" if sp["flags"] & 128 else ", holds its last frame") if frames > 1 else ""))
            for f in range(frames):
                if sp["kind"] == 4:
                    pos, at = sp["frames"][f * sp["stride"]], sp["frames"][f * sp["stride"] + 1]
                    kids = sp["childlist"][at + 1:at + 1 + sp["childlist"][at]]
                    lines.append(f"  frame {f}:")
                else:
                    pos, kids = 0, sp["children"]
                for kid in kids:
                    if pos >= len(sp["data"]) or len(lines) >= max_lines:
                        break
                    ops, pos = self._ops_text(sp["data"][pos], sp["data"], pos + 1)
                    lines.append(f"    {self._child_text(kid)}{'  ' + ops if ops else ''}")
                    if kid in self.sprites and kid not in seen:
                        seen.add(kid)
                        todo.append(kid)
            lines.append("")
        if todo or len(lines) >= max_lines:
            lines.append("... (cut short)")
        return lines

    def _child_text(self, kid: int) -> str:
        if kid in self.imports:
            return f'part "{self.imports[kid]}"'
        if kid in self.sprites:
            name = self.sprites[kid]["name"]
            return f"sprite {kid}" + (f' "{name}"' if name else "")
        shape = self.shapes.get(kid)
        return f"shape {kid} ({shape[4]}x{shape[5]})" if shape else f"? {kid}"

    def _ops_text(self, op: int, data: list, pos: int) -> tuple[str, int]:
        """What opcode `op` (read order as in _apply) does, as text."""
        def floats(table: list, i: int, n: int) -> str:
            return ", ".join(f"{v:g}" for v in (round(x, 3) for x in table[i:i + n]))
        out = []
        if op & 12 == 12:
            out.append(f"color = color * ({floats(self.colors, data[pos + 1], 4)}) + ({floats(self.colors, data[pos], 4)})")
            pos += 2
        elif op & 4:
            out.append(f"color *= ({floats(self.colors, data[pos], 4)})")
            pos += 1
        elif op & 8:
            out.append(f"color += ({floats(self.colors, data[pos], 4)})")
            pos += 1
        if op & 1:
            out.append(f"matrix({floats(self.matrices, data[pos], 4)})")
            pos += 1
        if op & 2:
            out.append(f"move({floats(self.translations, data[pos], 2)})")
            pos += 1
        return "  ".join(out), pos


IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, (1.0, 1.0, 1.0, 1.0))


def _quads(parts: list, scale: float) -> list:
    """(anm, shape, origin, width vector, height vector, rgba) per visible shape, in output pixels."""
    quads = []
    for anm, (u0, v0, u1, v1, w, h, ox, oy), (a, b, c, d, tx, ty, col) in parts:
        if w <= 0 or h <= 0 or col[3] <= 0.004:
            continue
        x0 = (ox * a + oy * c + tx) * scale
        y0 = (ox * b + oy * d + ty) * scale
        quads.append((anm, (u0, v0, u1, v1), (x0, y0), (a * w * scale, b * w * scale), (c * h * scale, d * h * scale),
                      col))
    return quads


def _bounds(quads: list) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for _anm, _uv, (x0, y0), (wx, wy), (hx, hy), _col in quads:
        xs += [x0, x0 + wx, x0 + hx, x0 + wx + hx]
        ys += [y0, y0 + wy, y0 + hy, y0 + wy + hy]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def draw(quads: list, atlases: dict, box: tuple[int, int, int, int]):
    """RGBA image of quads, covering box (x0, y0, x1, y1) in output pixels. atlases: {id(anm): PIL RGBA image}."""
    from PIL import Image
    bx0, by0, bx1, by1 = box
    canvas = Image.new("RGBA", (max(1, bx1 - bx0), max(1, by1 - by0)))
    for anm, (u0, v0, u1, v1), (x0, y0), (wx, wy), (hx, hy), col in quads:
        atlas = atlases.get(id(anm))
        if atlas is None:
            continue
        aw, ah = atlas.size
        crop = (int(round(u0 * aw)), int(round(v0 * ah)), int(round(u1 * aw)), int(round(v1 * ah)))
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            continue
        # The game puts the shape's v1 (its bottom edge in the atlas) at the quad's top corner: pieces are stored
        # upside down.
        piece = atlas.crop(crop).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if col != (1.0, 1.0, 1.0, 1.0):
            arr = np.asarray(piece, np.float32) * np.clip(np.array(col, np.float32), 0.0, 4.0)
            piece = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        pw, ph = piece.size
        # output (X, Y) = origin + s * wvec + t * hvec, s and t in 0..1 across the piece
        det = wx * hy - hx * wy
        if abs(det) < 1e-9:
            continue
        xs = [x0, x0 + wx, x0 + hx, x0 + wx + hx]
        ys = [y0, y0 + wy, y0 + hy, y0 + wy + hy]
        qx0, qy0 = max(bx0, int(np.floor(min(xs)))), max(by0, int(np.floor(min(ys))))
        qx1, qy1 = min(bx1, int(np.ceil(max(xs)))), min(by1, int(np.ceil(max(ys))))
        if qx1 <= qx0 or qy1 <= qy0:
            continue
        # s = ( hy*(X-x0) - hx*(Y-y0)) / det, t = (-wy*(X-x0) + wx*(Y-y0)) / det; piece pixel = (s*pw, t*ph).
        # PIL samples at pixel centres of the output region whose (0, 0) is (qx0, qy0).
        ox, oy = qx0 - x0, qy0 - y0
        sa, sb = hy / det * pw, -hx / det * pw
        ta, tb = -wy / det * ph, wx / det * ph
        coeffs = (sa, sb, sa * ox + sb * oy, ta, tb, ta * ox + tb * oy)
        layer = piece.transform((qx1 - qx0, qy1 - qy0), Image.Transform.AFFINE, coeffs, Image.Resampling.BILINEAR)
        canvas.alpha_composite(layer, (qx0 - bx0, qy0 - by0))
    return canvas


def split_direction(name: str) -> tuple[str, str]:
    """'5_AnimStatique' -> ('5', 'AnimStatique'); names without a direction -> ('', name)."""
    m = _DIR_NAME.match(name)
    return (m.group(1), m.group(2)) if m else ("", name)


def pick_direction(names: list[str]) -> str:
    """The full name to show first among one animation's directions."""
    return min(names, key=lambda n: DIRECTION_ORDER.find(split_direction(n)[0]) % 10)


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

class _LRU(OrderedDict):
    """Tiny LRU: move_to_end on hit, popitem(last=False) on overflow. Keeps the caches from thrashing when a
    session walks over hundreds of atlases (the old code cleared everything at the cap)."""

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


class WakfuSession(GameSession):
    def __init__(self, plugin: EnginePlugin, path: str) -> None:
        super().__init__(plugin, path)
        self.root = path
        self._zips: dict[str, zipfile.ZipFile] = {}
        self._zip_lock = threading.Lock()
        self._anms: _LRU = _LRU(64)
        self._atlases: _LRU = _LRU(ATLAS_CACHE)
        self._clips: _LRU = _LRU(CLIP_CACHE)
        self._packs: dict[str, Any] = {}
        self._pack_lock = threading.Lock()
        self._by_key: dict[Any, Asset] | None = None

    # ---- files
    def zip(self, jar: str) -> zipfile.ZipFile:
        z = self._zips.get(jar)
        if z is None:
            z = self._zips[jar] = zipfile.ZipFile(jar)
        return z

    def read(self, jar: str, name: str) -> bytes:
        with self._zip_lock:
            return self.zip(jar).read(name)

    def raw(self, asset: Asset) -> bytes:
        tag = asset.ref[0]
        if tag == "zip":
            return self.read(asset.ref[1], asset.ref[2])
        if tag == "pk":
            return self.audio(asset)[0]
        if tag == "anim":
            frames, length = self.sprite_frames(asset)
            images = [self.image(f) for _t, f in frames]
            return gif_bytes(images, length / max(1, len(images)))
        if tag in ("still", "frame"):
            out = io.BytesIO()
            self.image(asset).save(out, "PNG")
            return out.getvalue()
        raise ValueError(f"unknown asset {asset!r}")

    def close(self) -> None:
        with self._zip_lock:
            for z in self._zips.values():
                z.close()
            self._zips.clear()
        with self._pack_lock:
            for f in self._packs.values():
                f.close()
            self._packs.clear()
        self._anms.clear()
        self._atlases.clear()
        self._clips.clear()

    # ---- anm
    def anm(self, jar: str, name: str) -> Anm:
        key = (jar, name)
        a = self._anms.get_lru(key)
        if a is None:
            a = Anm(self.read(jar, name))
            self._anms.put(key, a)
        return a

    def atlas(self, jar: str, anm: Anm):
        key = (jar, anm.texture)
        img = self._atlases.get_lru(key)
        if img is None:
            from PIL import Image
            try:
                img = pil_image_from_bytes(self.read(jar, f"Atlas/{anm.texture}.png")).convert("RGBA")
            except (KeyError, OSError, ValueError):
                img = Image.new("RGBA", (1, 1))
            self._atlases.put(key, img)
        return img

    def _scale(self, jar: str, anm: Anm) -> float:
        """Draw at the atlas' own resolution (shapes often store a smaller size than their atlas rectangle), and
        at least twice the shape size: the game zooms in too, and tiny frames look poor in the viewer."""
        aw, ah = self.atlas(jar, anm).size
        ratios = sorted((u1 - u0) * aw / w for u0, _v0, u1, _v1, w, _h, _ox, _oy in anm.shapes.values() if w > 4)
        return float(min(MAX_SCALE, max(MIN_SCALE, ratios[len(ratios) // 2]))) if ratios else MIN_SCALE

    def clip(self, ref: tuple) -> tuple:
        """(per-frame quads, box, fps, atlases) of an animation ref ("anim", jar, .anm, sprite name, host .anm or "")
        (cached; all frames share one box so they line up)."""
        cached = self._clips.get_lru(ref)
        if cached is not None:
            return cached
        _tag, jar, name, sprite, host_name = ref
        anm = self.anm(jar, name)
        host = self.anm(jar, host_name) if host_name else None
        sid = anm.named.get(sprite)
        if sid is None:
            raise ValueError(f"{sprite} isn't in {name}")
        scale = self._scale(jar, anm)
        frames = []
        for f in range(anm.frame_count(sid)):
            parts: list = []
            anm.collect(sid, f, IDENTITY, parts, host)
            frames.append(_quads(parts, scale))
        atlases = {id(anm): self.atlas(jar, anm)}
        if host is not None:
            atlases[id(host)] = self.atlas(jar, host)
        bounds = [b for b in (_bounds(q) for q in frames) if b]
        if bounds:
            box = (int(np.floor(min(b[0] for b in bounds))) - 1, int(np.floor(min(b[1] for b in bounds))) - 1,
                   int(np.ceil(max(b[2] for b in bounds))) + 1, int(np.ceil(max(b[3] for b in bounds))) + 1)
        else:
            box = (0, 0, 1, 1)
        result = (frames, box, anm.fps, atlases)
        self._clips.put(ref, result)
        return result

    def _sheet(self, jar: str, name: str):
        """Still of an .anm without animations (equipment pieces, part libraries): its named parts side by side,
        with padding so adjacent tiles can't bleed into each other."""
        from PIL import Image
        anm = self.anm(jar, name)
        scale = self._scale(jar, anm)
        atlases = {id(anm): self.atlas(jar, anm)}
        tiles = []
        for sname, sid in sorted(anm.named.items())[:SHEET_TILE_LIMIT]:
            parts: list = []
            anm.collect(sid, 0, IDENTITY, parts)
            quads = _quads(parts, scale)
            b = _bounds(quads)
            if b:
                box = (int(np.floor(b[0])), int(np.floor(b[1])), int(np.ceil(b[2])) + 1, int(np.ceil(b[3])) + 1)
                tiles.append(draw(quads, atlases, box))
        if not tiles:
            return self.atlas(jar, anm)
        pad = 6
        cols = max(1, int(np.ceil(np.sqrt(len(tiles)))))
        cw = max(t.width for t in tiles) + pad
        rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
        heights = [max(t.height for t in row) + pad for row in rows]
        used_cols = max(len(row) for row in rows)
        sheet = Image.new("RGBA", (cw * used_cols + pad, sum(heights) + pad), (0, 0, 0, 0))
        y = pad
        for row, rh in zip(rows, heights):
            for i, t in enumerate(row):
                sheet.alpha_composite(t, (pad + i * cw + (cw - pad - t.width) // 2, y + rh - pad - t.height))
            y += rh
        return sheet

    # ---- GameSession
    def image(self, asset: Asset):
        tag = asset.ref[0]
        if tag == "frame":
            _tag, anim, index = asset.ref
            frames, box, _fps, atlases = self.clip(anim.ref)
            return draw(frames[index], atlases, box)
        if tag == "still":
            _tag, jar, name, anim = asset.ref
            if anim is None:
                return self._sheet(jar, name)
            frames, box, _fps, atlases = self.clip(anim)
            return draw(next((f for f in frames if f), frames[0]), atlases, box)
        data = self.raw(asset)
        if asset.ext == "tgam":
            return tgam_image(data)
        if asset.ext == "dds":
            return dds_image(data)
        return pil_image_from_bytes(data)

    def sprite_frames(self, asset: Asset):
        if asset.ref[0] != "anim":
            return [], 0.0
        frames, _box, fps, _atlases = self.clip(asset.ref)
        step = 1.0 / fps
        out = []
        for i in range(len(frames)):
            out.append((i * step, Asset("sprite", f"{asset.name} #{i}", ("frame", asset.key, i),
                                        uid=f"{asset.uid}#{i}", source=asset.source, ref=("frame", asset, i))))
        return out, len(frames) * step

    def animation_dim(self, asset):
        return "2d"  # every animation here is a flipbook

    def sprite_regions(self, asset: Asset):
        if asset.ref[0] != "frame" or asset.ref[1].ref[0] != "anim":
            return []
        _tag, anim, index = asset.ref
        frames, (bx, by, _bx1, _by1), _fps, atlases = self.clip(anim.ref)
        rel_jar = self._rel(anim.ref[1])
        if self._by_key is None:
            self._by_key = {a.key: a for a in self.assets}
        out: list[tuple[Asset, tuple[float, float, float, float], Any]] = []
        for anm, (u0, v0, u1, v1), (x0, y0), (wx, wy), (hx, hy), _col in frames[index]:
            sheet = self._by_key.get(("zip", rel_jar, f"Atlas/{anm.texture}.png"))
            atlas = atlases.get(id(anm))
            if sheet is None or atlas is None:
                continue
            aw, ah = atlas.size
            left, right, top, bottom = round(u0 * aw), round(u1 * aw), round(v0 * ah), round(v1 * ah)
            x, y = x0 - bx, y0 - by  # image(frame) covers the clip's box
            # Pieces are stored upside down (see draw): the region's bottom-left lands on the quad's origin.
            corners = [(x + hx, y + hy), (x + wx + hx, y + wy + hy), (x + wx, y + wy), (x, y)]
            out.append((sheet, (float(left), float(ah - bottom), float(right - left), float(bottom - top)), corners))
        return out

    def text(self, asset: Asset) -> str | bytes:
        if asset.ref[0] == "anim":
            _tag, jar, name, sprite, host = asset.ref
            anm = self.anm(jar, name)
            sid = anm.named[sprite]
            n = anm.frame_count(sid)
            base = split_direction(sprite)[1]
            dirs = sorted(d for d, b in (split_direction(s) for s in anm.named) if b == base and d)
            lines = [asset.name, "", f"{n} frames at {anm.fps} fps ({n / anm.fps:.2f} s), from {name}"
                     + (f" with the parts of {host}" if host else "") + ".",
                     f"Shown: {sprite}" + (f" (directions in the file: {', '.join(dirs)})" if dirs else ""),
                     "Export saves it as an animated GIF."]
            return "\n".join(lines)
        return super().text(asset)

    def _anm_files(self, asset: Asset) -> tuple[str, str, list[str]] | None:
        """(jar, rel_jar, [.anm files drawn]) of an animation, a still or one of their frames; None for others."""
        tag = asset.ref[0]
        if tag == "frame":
            asset = asset.ref[1]
            tag = asset.ref[0]
        if tag == "anim":
            _tag, jar, name, _sprite, host = asset.ref
            return jar, self._rel(jar), [name] + ([host] if host and host != name else [])
        if tag == "still":
            _tag, jar, name, first = asset.ref
            host = first[4] if first else ""
            return jar, self._rel(jar), [name] + ([host] if host and host != name else [])
        return None

    def _rel(self, jar: str) -> str:
        return os.path.relpath(jar, os.path.join(self.root, "contents")).replace("\\", "/")

    def related(self, asset: Asset):
        found = self._anm_files(asset)
        if found is None:
            return super().related(asset)
        jar, rel_jar, names = found
        if self._by_key is None:
            self._by_key = {a.key: a for a in self.assets}
        sheets: list[Asset] = []
        for name in names:
            try:
                texture = self.anm(jar, name).texture
            except Exception:
                continue
            sheet = self._by_key.get(("zip", rel_jar, f"Atlas/{texture}.png"))
            if sheet is not None and sheet not in sheets:
                sheets.append(sheet)
        return "Sprite sheet", sheets, "No atlas image found for this animation."

    def animation_code(self, asset: Asset) -> tuple[str, str]:
        if asset.ref[0] != "anim":
            return super().animation_code(asset)
        _tag, jar, name, sprite, host = asset.ref
        anm = self.anm(jar, name)
        sid = anm.named[sprite]
        lines = [f"{name} -> {sprite} (sprite {sid}), {anm.fps} fps" + (f", parts from {host}" if host else ""),
                 "How it plays: every frame walks this sprite tree. Each child line is one opcode: color (multiply / add),",
                 "matrix (a, b, c, d: x' = a x + c y, y' = b x + d y) and move (x, y), applied on top of the",
                 "parent's transform. No opcode = the parent's transform as is. Nested sprites play on the",
                 "same clock; past their end they hold the last frame unless they loop.",
                 "Shapes are rectangles of the sprite sheet; parts come from the host .anm.", ""]
        return f"Animation code ({sprite})", "\n".join(lines + anm.code(sid))

    def audio(self, asset: Asset):
        if asset.ref[0] == "pk":
            _tag, pack, _sid, off, size = asset.ref
            with self._pack_lock:
                f = self._packs.get(pack)
                if f is None:
                    f = self._packs[pack] = open(pack, "rb")
                f.seek(off)
                data = f.read(size)
            return pk_ogg(data), "ogg"
        return super().audio(asset)

    def describe(self, asset: Asset) -> list[tuple[str, str]]:
        rows = super().describe(asset)
        tag = asset.ref[0]
        if tag in ("anim", "still"):
            jar, name = asset.ref[1], asset.ref[2]
            try:
                anm = self.anm(jar, name)
                rows += [("Atlas", f"Atlas/{anm.texture}.png"), ("Shapes", f"{len(anm.shapes):,}"),
                         ("Sprites", f"{len(anm.sprites):,} ({len(anm.named):,} named)")]
                if anm.imports:
                    rows.append(("Imported parts", f"{len(anm.imports):,}"))
                if anm.files:
                    rows.append(("Animation files", ", ".join(anm.files[:12]) + (" ..." if len(anm.files) > 12 else "")))
            except Exception as e:
                rows.append(("Error", str(e)))
        return rows

    def stats(self, asset: Asset) -> AssetStats:
        stats: AssetStats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        tag = asset.ref[0]
        try:
            if tag == "anim":
                _tag, jar, name, sprite, _host = asset.ref
                anm = self.anm(jar, name)
                n = anm.frame_count(anm.named[sprite])
                stats["info"], stats["sort"] = f"{n} frames", n
            elif asset.ext == "tgam":
                head = self.read(asset.ref[1], asset.ref[2])[:8]
                w, h = tgam_size(head)
                stats["w"], stats["h"], stats["info"], stats["sort"] = w, h, f"{w}×{h}", w * h
            elif asset.ext == "dds":
                data = self.read(asset.ref[1], asset.ref[2])[:16]
                if len(data) >= 16:
                    h, w = struct.unpack_from("<II", data, 12)
                    stats["w"], stats["h"], stats["info"], stats["sort"] = w, h, f"{w}×{h}", w * h
        except Exception:
            pass
        return stats


class WakfuPlugin(EnginePlugin):
    id = "wakfu"
    name = "Wakfu (Ankama)"
    version = "1.2"
    author = "UniView"
    description = ("Wakfu: animations, map graphics, UI images (incl. DDS), sounds and texts from contents/*.jar "
                   "and the .pk packs.")

    @staticmethod
    def _root(path: str) -> str | None:
        path = os.path.normpath(path)
        for p in (path, os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
            if os.path.isdir(os.path.join(p, "contents", "animations")) and \
                    os.path.isfile(os.path.join(p, "contents", "gui_jar", "gui.jar")):
                return p
        return None

    def detect(self, path: str) -> int:
        root = self._root(path)
        if root is None:
            return 0
        return 95 if os.path.isfile(os.path.join(root, "lib", "wakfu-client.jar")) else 70

    def game_info(self, path: str) -> GameInfo:
        return {"engine_version": "Ankama", "detail": "Wakfu"}

    def _jars(self, root: str) -> list[str]:
        contents = os.path.join(root, "contents")
        jars = []
        for d, dirs, files in os.walk(contents):
            rel = os.path.relpath(d, contents).replace("\\", "/")
            if rel.split("/")[0] in SKIP_DIRS:
                dirs[:] = []
                continue
            jars += [os.path.join(d, f) for f in sorted(files) if f.endswith(".jar")]
        gfx = os.path.join(contents, "maps", "gfx.jar")
        if os.path.isfile(gfx):
            jars.append(gfx)
        return jars

    def _packs(self, root: str) -> list[str]:
        sounds = os.path.join(root, "contents", "sounds")
        try:
            groups = sorted(os.listdir(sounds))
        except OSError:
            return []
        return [p for g in groups for p in [os.path.join(sounds, g, g + ".pk")] if os.path.isfile(p)]

    def count_files(self, path: str) -> int:
        root = self._root(path) or path
        return len(self._jars(root)) + len(self._packs(root))

    def open(self, path: str, progress) -> WakfuSession:
        root = self._root(path)
        if root is None:
            raise FileNotFoundError(f"No Wakfu contents/ folder found in:\n{path}")
        session = WakfuSession(self, root)
        contents = os.path.join(root, "contents")
        jars, packs = self._jars(root), self._packs(root)
        progress.files([(os.path.relpath(j, contents), os.path.getsize(j)) for j in jars + packs])
        # Reading: every jar's file list (+ the start of each .anm) and every sound pack's table.
        listings = []
        for i, jar in enumerate(jars):
            progress.file(i)
            rel_jar = os.path.relpath(jar, contents).replace("\\", "/")
            progress(f"Reading {rel_jar} ...", i, len(jars) + len(packs))
            try:
                with zipfile.ZipFile(jar) as z:
                    infos = [info for info in z.infolist()
                             if not info.is_dir() and not _is_skippable(info.filename)]
                    anms = [info for info in infos if info.filename.endswith(".anm")]
                    listings.append((jar, rel_jar, infos, anms, self._anm_indexes(z, anms)))
            except (OSError, zipfile.BadZipFile) as e:
                session.warnings.append(f"{rel_jar}: {e}")
                listings.append((jar, rel_jar, [], [], {}))
        pack_entries = []
        for i, pack in enumerate(packs):
            progress.file(len(jars) + i)
            progress(f"Reading sounds/{os.path.basename(pack)} ...", len(jars) + i, len(jars) + len(packs))
            try:
                with open(pack, "rb") as f:
                    pack_entries.append((pack, pk_entries(f)))
            except OSError as e:
                session.warnings.append(f"{os.path.basename(pack)}: {e}")
                pack_entries.append((pack, []))
        progress.file(len(jars) + len(packs))

        # Indexing: the asset list.
        found: dict[str, int] = {}

        def add(asset: Asset) -> None:
            session.assets.append(asset)
            found[asset.kind] = found.get(asset.kind, 0) + 1

        steps = len(listings) + len(pack_entries)
        for i, (jar, rel_jar, infos, anms, indexes) in enumerate(listings):
            progress(f"Indexing {rel_jar} ...", i, steps)
            folder = os.path.dirname(rel_jar)
            if anms:
                self._add_anms(jar, rel_jar, anms, indexes, add, session)
            for info in infos:
                name = info.filename
                ext = os.path.splitext(name)[1].lower().lstrip(".")
                if ext == "anm":
                    continue
                stem = name[:-len(ext) - 1] if ext else name
                kind: Kind
                if ext in IMAGE_EXTS or ext == "tgam":
                    kind = "texture"
                elif ext in TEXT_EXTS:
                    kind = "text"
                else:
                    kind = kind_for_extension(ext)
                label = f"{folder}/{stem}" if kind == "texture" else f"{folder}/{name}"
                add(Asset(kind=kind, name=label, key=("zip", rel_jar, name), uid=f"wakfu:{rel_jar}:{name}",
                          size=info.file_size, path=f"{folder}/{name}", source=rel_jar, ref=("zip", jar, name),
                          ext=ext))
            progress.found(found)
        for i, (pack, entries) in enumerate(pack_entries):
            group = os.path.splitext(os.path.basename(pack))[0]
            progress(f"Indexing sounds/{group} ...", len(listings) + i, steps)
            for sid, off, size in entries:
                add(Asset(kind="audio", name=f"sounds/{group}/{sid}", key=("pk", group, sid),
                          uid=f"wakfu:pk:{group}:{sid}", size=size, path=f"sounds/{group}/{sid}.ogg",
                          source=f"sounds/{group}/{group}.pk", ref=("pk", pack, sid, off, size), ext="ogg"))
            progress.found(found)
        session.file_count = len(jars) + len(packs)
        return session

    @staticmethod
    def _anm_indexes(z: zipfile.ZipFile, anms: list) -> dict[str, tuple[list[str], list[tuple[str, int]]]]:
        """{.anm name: (other .anm files, [(animation, file index)])}, read from the start of each file. One file
        that won't parse doesn't take down the whole jar: it's recorded as having no animations."""
        indexes: dict[str, tuple[list[str], list[tuple[str, int]]]] = {}
        for info in anms:
            try:
                with z.open(info) as f:
                    head = f.read(8192)
            except (OSError, zipfile.BadZipFile):
                head = b""
            try:
                indexes[info.filename] = anm_index(head)
                continue
            except (struct.error, ValueError, IndexError):
                pass
            try:
                indexes[info.filename] = anm_index(z.read(info))
            except (struct.error, ValueError, IndexError, OSError, zipfile.BadZipFile):
                indexes[info.filename] = ([], [])
        return indexes

    @staticmethod
    def _add_anms(jar: str, rel_jar: str, anms: list, indexes: dict, add,
                  session: "WakfuSession") -> None:
        """An animation per animation name (directions folded together) and a still per .anm."""
        folder = os.path.dirname(rel_jar)
        group = os.path.basename(folder)
        hosts: dict[str, str] = {}  # shared animation file -> the first .anm that lists it
        for name, (files, _anims) in indexes.items():
            for other in files:
                hosts.setdefault(other, name)
        for info in anms:
            name = info.filename
            stem = name[:-4]
            shown = stem
            host = hosts.get(name, "")
            files, anims = indexes.get(name, ([], []))
            by_base: dict[str, list[str]] = {}
            for anim, file_index in anims:
                if file_index < 0:  # the others live in (and get listed with) files[file_index]
                    by_base.setdefault(split_direction(anim)[1], []).append(anim)
            first: tuple | None = None
            for base in sorted(by_base, key=str.lower):
                sprite = pick_direction(by_base[base])
                key = ("anim", rel_jar, name, base)
                asset = Asset(kind="animation", name=f"{group}/{shown}/{base}", key=key,
                              uid=f"wakfu:{rel_jar}:{name}:{base}", size=None, path=f"{folder}/{shown}/{base}.gif",
                              source=rel_jar, ref=("anim", jar, name, sprite, host), ext="gif")
                add(asset)
                if first is None or "statique" in base.lower() and "statique" not in first[3].lower():
                    first = asset.ref
            if first is None:  # a player class: its animations are in shared files that use its parts
                outside = [(anim, files[i]) for anim, i in anims if 0 <= i < len(files)]
                pick = [o for o in outside if split_direction(o[0])[1] == "AnimStatique"] or outside
                if pick:
                    anim = pick_direction([o[0] for o in pick])
                    first = ("anim", jar, dict(pick)[anim], anim, name)
            # A still for anything with named parts; empty .anm files are skipped so the list stays useful.
            if not by_base and first is None:
                try:
                    has_named = bool(session.anm(jar, name).named)
                except Exception:
                    has_named = False
                if not has_named and not anims:
                    continue
            add(Asset(kind="sprite", name=f"{group}/{shown}", key=("still", rel_jar, name),
                      uid=f"wakfu:{rel_jar}:{name}", size=info.file_size, path=f"{folder}/{shown}.png",
                      source=rel_jar, ref=("still", jar, name, first), ext="png"))


def _is_skippable(name: str) -> bool:
    """Jar bookkeeping that isn't an asset."""
    if any(name.startswith(p) for p in SKIP_PREFIXES):
        return True
    low = name.lower()
    return low.endswith(SKIP_SUFFIXES) or os.path.basename(low) in ("desktop.ini", "manifest.mf", "index.list")


PLUGIN = WakfuPlugin()