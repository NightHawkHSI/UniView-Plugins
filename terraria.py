"""Terraria (and other XNA / FNA games: Stardew Valley, Celeste...) - a complete engine plugin for UniView.

Drop this file into UniView's plugins/ folder. It is also the worked example of a real plugin: everything is in
this one file and only uses the standard library, numpy and Pillow (what the packaged UniView.exe has), plus the
plugin API in engines/sdk.py.

What it reads from the game's Content folder:
  * .xnb files - XNA's content format. Textures (Images/...) and sprite-font sheets show as textures, SoundEffect
    files (Sounds/...) as sounds; shaders and game data are listed for raw export.
    Layout: "XNB" + platform + version 5 + flags (0x80 = LZX compressed, 0x40 = HiDef) + total size; a compressed
    file then has the decompressed size and LZX data in frames (XMemCompress, 64 KB window). The content is a list
    of type readers, then the object written by its reader (Texture2D: format, size, mip levels; SoundEffect: a
    WAVEFORMATEX and the sample data; sprite fonts: their glyph texture inside).
  * XACT wave banks (.xwb, Terraria's music): five segments (bank data, entry metadata, seek tables, names, wave
    data); each entry's format packs codec / channels / rate / block align into one 32-bit word. Terraria's music
    is MS-ADPCM, decoded here with numpy.
  * XACT sound banks (.xsb): the cue names the game plays music by ("Music_1"...), used to name the wave entries.
  * Terraria.exe (plain .NET): the `const` fields of Terraria.ID.ItemID, NPCID, TileID... give id -> internal
    name, its embedded en-US localization JSON the display names: Images/Item_1 -> "Item: Iron Pickaxe (1)".
    Its IL gives the frame counts (Main.npcFrameCount, Main.projFrames, RegisterItemAnimation calls): animated
    sheets get an "Animations/..." entry (flipbook in the viewer, GIF on export) and sheets show their frames.
    MapHelper.Initialize's IL gives the map colors of tiles and walls.
    Item / NPC / projectile stats come from their SetDefaults methods (if-chains and switch tables of
    `this.damage = 12`), the ID sets (ItemID.Sets...) from their static constructors and tile flags
    from Main.tileSolid[...] = true: all shown in the info panel.
  * Saves (Documents/My Games/Terraria and Steam Cloud): worlds (.wld: header + run-length tiles) drawn as the
    full map, players (.plr: AES-128-CBC with the game's fixed key) as text - stats, equipment, inventory, banks.

Sections: LZX decompression, XNB reading, MS-ADPCM, XACT banks, .NET metadata, Terraria names, then the plugin
(XnaSession / XnaPlugin).
"""

import io
import json
import logging
import os
import re
import struct

import numpy as np

from engines.sdk import Asset, EnginePlugin, GameSession, listdir_lower

log = logging.getLogger("viewer.plugin.terraria")

# --------------------------------------------------------------------------- LZX (XNA flavour)

MIN_MATCH = 2
VERBATIM, ALIGNED, UNCOMPRESSED = 1, 2, 3

EXTRA_BITS, POSITION_BASE = [], []
_j = 0
for _i in range(0, 52, 2):
    EXTRA_BITS += [_j, _j]
    if _i != 0 and _j < 17:
        _j += 1
_base = 0
for _bits in EXTRA_BITS:
    POSITION_BASE.append(_base)
    _base += 1 << _bits


def _table(lengths):
    """Canonical Huffman decode table: (list of symbols by every `bits`-bit prefix, bits = the longest code)."""
    bits = max(max(lengths), 1)
    table = [0] * (1 << bits)
    pos = 0
    by_len: dict[int, list[int]] = {}
    for sym, n in enumerate(lengths):
        if n:
            by_len.setdefault(n, []).append(sym)
    for n in sorted(by_len):
        span = 1 << (bits - n)
        for sym in by_len[n]:
            if pos + span > len(table):
                raise ValueError("LZX: bad Huffman table")
            table[pos:pos + span] = [sym] * span
            pos += span
    return table, bits


class _Bits:
    """16-bit little-endian words, read most significant bit first."""

    __slots__ = ("d", "p", "buf", "n")

    def __init__(self, data, pos=0):
        self.d, self.p, self.buf, self.n = data, pos, 0, 0

    def need(self, k):
        d = self.d
        while self.n < k:
            p = self.p
            w = (d[p] | (d[p + 1] << 8)) if p + 1 < len(d) else 0
            self.p = p + 2
            self.buf = (self.buf << 16) | w
            self.n += 16

    def read(self, k):
        if k == 0:
            return 0
        if self.n < k:
            self.need(k)
        self.n -= k
        v = self.buf >> self.n
        self.buf &= (1 << self.n) - 1
        return v

    def sym(self, huffman, lengths):
        table, bits = huffman
        if self.n < bits:
            self.need(bits)
        s = table[self.buf >> (self.n - bits)]
        self.n -= lengths[s]
        self.buf &= (1 << self.n) - 1
        return s


class LzxDecoder:
    """LZX decompressor state, kept across the frames of one file (port of MonoGame's LzxDecoder)."""

    def __init__(self, window_bits=16):
        self.size = 1 << window_bits
        self.window = bytearray(self.size)
        self.posn = 0
        self.r0 = self.r1 = self.r2 = 1
        slots = {15: 30, 16: 32, 17: 34, 18: 36, 19: 38, 20: 42, 21: 50}[window_bits]
        self.main_len = [0] * (256 + slots * 8)
        self.length_len = [0] * 249
        self.aligned_len = [0] * 8
        self.main_table = self.length_table = self.aligned_table = None
        self.header_read = False
        self.block_type = self.block_remaining = self.block_length = 0

    def decompress(self, data, out_len):
        bits = _Bits(data)
        window, size = self.window, self.size
        if not self.header_read:
            if bits.read(1):
                bits.read(32)  # Intel E8 file size: XNA content doesn't use the translation
            self.header_read = True
        togo = out_len
        while togo > 0:
            if self.block_remaining == 0:
                if self.block_type == UNCOMPRESSED:
                    if self.block_length & 1:
                        bits.p += 1
                    bits.buf = bits.n = 0
                self.block_type = bits.read(3)
                self.block_remaining = self.block_length = (bits.read(16) << 8) | bits.read(8)
                if self.block_type == ALIGNED:
                    self.aligned_len = [bits.read(3) for _ in range(8)]
                    self.aligned_table = _table(self.aligned_len)
                if self.block_type in (VERBATIM, ALIGNED):
                    self._lengths(bits, self.main_len, 0, 256)
                    self._lengths(bits, self.main_len, 256, len(self.main_len))
                    self.main_table = _table(self.main_len)
                    self._lengths(bits, self.length_len, 0, 249)
                    self.length_table = _table(self.length_len)
                elif self.block_type == UNCOMPRESSED:
                    bits.need(16)
                    if bits.n > 16:
                        bits.p -= 2
                    bits.buf = bits.n = 0
                    self.r0, self.r1, self.r2 = struct.unpack_from("<3I", data, bits.p)
                    bits.p += 12
                else:
                    raise ValueError(f"LZX: unknown block type {self.block_type}")
            this_run = min(self.block_remaining, togo)
            togo -= this_run
            self.block_remaining -= this_run
            self.posn &= size - 1
            if self.posn + this_run > size:
                raise ValueError("LZX: run past the window")
            if self.block_type == UNCOMPRESSED:
                window[self.posn:self.posn + this_run] = data[bits.p:bits.p + this_run]
                bits.p += this_run
                self.posn += this_run
                continue
            self._run(bits, this_run)
        end = self.posn or size
        return bytes(window[end - out_len:end])

    def _lengths(self, bits, lens, first, last):
        pre = [bits.read(4) for _ in range(20)]
        table = _table(pre)
        x = first
        while x < last:
            z = bits.sym(table, pre)
            if z == 17:
                y = min(bits.read(4) + 4, last - x)
                lens[x:x + y] = [0] * y
            elif z == 18:
                y = min(bits.read(5) + 20, last - x)
                lens[x:x + y] = [0] * y
            elif z == 19:
                y = min(bits.read(1) + 4, last - x)
                z = (lens[x] - bits.sym(table, pre)) % 17
                lens[x:x + y] = [z] * y
            else:
                lens[x] = (lens[x] - z) % 17
                y = 1
            x += y

    def _run(self, bits, this_run):
        window, size = self.window, self.size
        main_table, main_len = self.main_table, self.main_len
        length_table, length_len = self.length_table, self.length_len
        aligned = self.block_type == ALIGNED
        aligned_table, aligned_len = self.aligned_table, self.aligned_len
        r0, r1, r2 = self.r0, self.r1, self.r2
        posn = self.posn
        while this_run > 0:
            main = bits.sym(main_table, main_len)
            if main < 256:
                window[posn] = main
                posn += 1
                this_run -= 1
                continue
            main -= 256
            length = main & 7
            if length == 7:
                length += bits.sym(length_table, length_len)
            length += MIN_MATCH
            slot = main >> 3
            if slot > 2:
                extra = EXTRA_BITS[slot]
                offset = POSITION_BASE[slot] - 2
                if not aligned:
                    offset += bits.read(extra)
                elif extra > 3:
                    offset += (bits.read(extra - 3) << 3) + bits.sym(aligned_table, aligned_len)
                elif extra == 3:
                    offset += bits.sym(aligned_table, aligned_len)
                elif extra > 0:
                    offset += bits.read(extra)
                else:
                    offset = 1
                r2, r1, r0 = r1, r0, offset
            elif slot == 0:
                offset = r0
            elif slot == 1:
                offset = r1
                r1, r0 = r0, offset
            else:
                offset = r2
                r2, r0 = r0, offset
            this_run -= length
            src = posn - offset
            if src < 0:
                src += size
            if src + length <= posn or (src >= posn + length and src + length <= size):
                window[posn:posn + length] = window[src:src + length]  # no overlap, no wrap
                posn += length
            else:
                for _ in range(length):
                    window[posn] = window[src]
                    posn += 1
                    src = (src + 1) & (size - 1)
        self.r0, self.r1, self.r2 = r0, r1, r2
        self.posn = posn


def lzx_frames(data, pos, out_size):
    """Decompress an XNB's LZX payload (frames from `pos` on) to out_size bytes."""
    dec = LzxDecoder(16)
    out = bytearray()
    end = len(data)
    while pos < end and len(out) < out_size:
        if data[pos] == 0xFF:
            frame = (data[pos + 1] << 8) | data[pos + 2]
            block = (data[pos + 3] << 8) | data[pos + 4]
            pos += 5
        else:
            frame, block = 0x8000, (data[pos] << 8) | data[pos + 1]
            pos += 2
        if block == 0 or frame == 0:
            break
        out += dec.decompress(data[pos:pos + block], frame)
        pos += block
    return bytes(out[:out_size])


# --------------------------------------------------------------------------- XNB

class XnbError(ValueError):
    pass


def xnb_header(data):
    """(platform, version, compressed, hidef) of an .xnb, or raise XnbError."""
    if len(data) < 10 or data[:3] != b"XNB":
        raise XnbError("Not an XNB file")
    flags = data[5]
    return chr(data[3]), data[4], bool(flags & 0x80), bool(flags & 0x40)


def xnb_content(data, limit=None):
    """The content bytes of an .xnb (decompressed when needed); limit: only (at least) the first this many."""
    _platform, _version, compressed, _hidef = xnb_header(data)
    if not compressed:
        return data[10:]
    size = struct.unpack_from("<I", data, 10)[0]
    return lzx_frames(data, 14, size if limit is None else min(size, limit))


class _Reader:
    def __init__(self, data):
        self.d, self.p = data, 0

    def u8(self):
        self.p += 1
        return self.d[self.p - 1]

    def i32(self):
        self.p += 4
        return struct.unpack_from("<i", self.d, self.p - 4)[0]

    def u32(self):
        self.p += 4
        return struct.unpack_from("<I", self.d, self.p - 4)[0]

    def bytes(self, n):
        self.p += n
        return self.d[self.p - n:self.p]

    def int7(self):
        out = shift = 0
        while True:
            b = self.u8()
            out |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return out

    def string(self):
        return self.bytes(self.int7()).decode("utf-8", "replace")


def xnb_object(content):
    """(type reader names, primary reader name, reader positioned at the object's data)."""
    r = _Reader(content)
    readers = []
    for _ in range(r.int7()):
        readers.append(r.string())
        r.i32()
    r.int7()  # shared resources
    index = r.int7()
    if not 0 < index <= len(readers):
        raise XnbError("Empty XNB")
    return readers, _short(readers[index - 1]), r


def _short(reader):
    """'Microsoft.Xna.Framework.Content.Texture2DReader, Microsoft.Xna.Framework...' -> 'Texture2DReader'."""
    name = reader.split(",", 1)[0]
    return name.rsplit(".", 1)[-1].split("`", 1)[0]


def xnb_kind(data):
    """The reader of the main object ('Texture2DReader', 'SoundEffectReader', 'SpriteFontReader', ...)."""
    return xnb_object(xnb_content(data))[1]


# SurfaceFormat values (XNA 4)
COLOR, BGR565, BGRA5551, BGRA4444, DXT1, DXT3, DXT5 = 0, 1, 2, 3, 4, 5, 6


def _texture(r):
    """(PIL image, surface format) of a Texture2D at the reader (its first mip level)."""
    from PIL import Image
    fmt, w, h, levels = r.i32(), r.u32(), r.u32(), r.u32()
    if levels < 1:
        raise XnbError("Texture without data")
    size = r.u32()
    pixels = r.bytes(size)
    if fmt == COLOR:
        img = Image.frombytes("RGBA", (w, h), bytes(pixels[:w * h * 4]))
    elif fmt in (DXT1, DXT3, DXT5):
        img = Image.frombytes("RGBA", (w, h), bytes(pixels), "bcn", {DXT1: 1, DXT3: 2, DXT5: 3}[fmt])
    elif fmt == BGR565:
        v = np.frombuffer(pixels, "<u2", w * h).reshape(h, w).astype(np.uint32)
        rgb = np.stack([(v >> 11) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31,
                        np.full_like(v, 255)], -1).astype(np.uint8)
        img = Image.fromarray(rgb)
    elif fmt == BGRA4444:
        v = np.frombuffer(pixels, "<u2", w * h).reshape(h, w)
        rgba = np.stack([(v >> 8) & 15, (v >> 4) & 15, v & 15, v >> 12], -1).astype(np.uint8) * 17
        img = Image.fromarray(rgba)
    elif fmt == BGRA5551:
        v = np.frombuffer(pixels, "<u2", w * h).reshape(h, w).astype(np.uint32)
        rgba = np.stack([((v >> 10) & 31) * 255 // 31, ((v >> 5) & 31) * 255 // 31, (v & 31) * 255 // 31,
                         (v >> 15) * 255], -1).astype(np.uint8)
        img = Image.fromarray(rgba)
    else:
        raise XnbError(f"Texture format {fmt} isn't supported")
    return img, fmt


def unpremultiply(img):
    """XNA's content pipeline stores colors premultiplied by alpha; undo it for viewing / PNG export."""
    a = np.asarray(img.convert("RGBA")).copy()
    alpha = a[..., 3:4].astype(np.uint16)
    if not (alpha < 255).any() or (a[..., :3] > alpha).any():
        return img  # opaque, or not premultiplied after all
    rgb = np.where(alpha > 0, np.minimum(a[..., :3].astype(np.uint16) * 255 // np.maximum(alpha, 1), 255), 0)
    a[..., :3] = rgb.astype(np.uint8)
    from PIL import Image
    return Image.fromarray(a)


MIP_SIZE = {COLOR: 4, BGR565: 2, BGRA5551: 2, BGRA4444: 2}


def _embedded_texture(readers, r):
    """Move the reader to the first Texture2D inside another object (a sprite font's glyph sheet, ReLogic's
    DynamicSpriteFont pages...): its type id followed by a plausible texture header."""
    if "Texture2DReader" not in [_short(n) for n in readers]:
        raise XnbError("No texture in this file")
    type_id = [_short(n) for n in readers].index("Texture2DReader") + 1
    d = r.d
    for p in range(r.p, len(d) - 21):
        if d[p] != type_id:
            continue
        fmt, w, h, levels, size = struct.unpack_from("<iIIII", d, p + 1)
        if not (0 <= fmt <= DXT5 and 0 < w <= 16384 and 0 < h <= 16384 and 0 < levels <= 15):
            continue
        expected = w * h * MIP_SIZE[fmt] if fmt in MIP_SIZE else max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * (
            8 if fmt == DXT1 else 16)
        if size == expected and p + 21 + size <= len(d):
            r.p = p + 1
            return
    raise XnbError("Couldn't find the texture in this file")


def _texture_reader(data):
    readers, main, r = xnb_object(xnb_content(data))
    if main != "Texture2DReader":
        _embedded_texture(readers, r)
    return r


def xnb_image(data):
    """The texture of a Texture2D .xnb - or the first one inside another object, like a sprite font's glyph
    sheet - as a PIL image (straight alpha)."""
    img, _fmt = _texture(_texture_reader(data))
    return unpremultiply(img)


def texture_info(data):
    """(width, height, surface format) of xnb_image()'s texture without decoding the pixels (a plain texture's
    header is in the first LZX frame; anything else is read whole)."""
    try:
        readers, main, r = xnb_object(xnb_content(data, 1024))
        if main != "Texture2DReader":
            raise IndexError
    except (IndexError, struct.error):
        r = _texture_reader(data)
    fmt, w, h = r.i32(), r.u32(), r.u32()
    return w, h, fmt


def wav_file(fmt, samples):
    """RIFF WAVE bytes from a WAVEFORMATEX (bytes) and the sample data."""
    fmt = bytes(fmt)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(samples)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + bytes(samples))


def xnb_sound(data):
    """WAV bytes of a SoundEffect .xnb (PCM or MS-ADPCM, decoded to PCM)."""
    _readers, main, r = xnb_object(xnb_content(data))
    if main != "SoundEffectReader":
        raise XnbError(f"Not a sound ({main})")
    fmt = r.bytes(r.u32())
    samples = r.bytes(r.u32())
    tag, channels, rate = struct.unpack_from("<HHI", fmt, 0)
    if tag == 2:  # MS-ADPCM
        block_align = struct.unpack_from("<H", fmt, 12)[0]
        pcm = msadpcm_decode(samples, channels, block_align)
        return pcm_wav(pcm, channels, rate)
    return wav_file(fmt, samples)


def pcm_wav(pcm, channels, rate):
    """WAV bytes of interleaved int16 samples."""
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
    return wav_file(fmt, np.ascontiguousarray(pcm, "<i2").tobytes())


# --------------------------------------------------------------------------- MS-ADPCM

ADAPT = np.array([230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230], np.int64)
COEF1 = np.array([256, 512, 0, 192, 240, 460, 392], np.int64)
COEF2 = np.array([0, -256, 0, 64, 0, -208, -232], np.int64)


def msadpcm_decode(data, channels, block_align):
    """Interleaved int16 samples of MS-ADPCM data. Every block starts from its own header, so all blocks are
    decoded together (numpy across blocks, a Python loop only over the samples of one block)."""
    data = bytes(data)
    nblocks = len(data) // block_align
    if nblocks == 0:
        return np.zeros(0, np.int16)
    blocks = np.frombuffer(data[:nblocks * block_align], np.uint8).reshape(nblocks, block_align)
    c = channels
    head = 7 * c
    pred = blocks[:, :c].astype(np.int64).clip(0, 6)
    words = blocks[:, c:head].copy().view("<i2").reshape(nblocks, 3, c).astype(np.int64)
    delta, s1, s2 = words[:, 0], words[:, 1], words[:, 2]
    c1, c2 = COEF1[pred], COEF2[pred]
    nib = blocks[:, head:]
    nibbles = np.empty((nblocks, nib.shape[1] * 2), np.int64)
    nibbles[:, 0::2] = nib >> 4
    nibbles[:, 1::2] = nib & 15
    per_channel = nibbles.shape[1] // c
    out = np.empty((nblocks, per_channel + 2, c), np.int64)
    out[:, 0], out[:, 1] = s2, s1
    nibbles = nibbles[:, :per_channel * c].reshape(nblocks, per_channel, c)
    for i in range(per_channel):
        n = nibbles[:, i]
        signed = np.where(n >= 8, n - 16, n)
        predict = (s1 * c1 + s2 * c2) >> 8
        sample = np.clip(predict + signed * delta, -32768, 32767)
        s2, s1 = s1, sample
        delta = np.maximum((ADAPT[n] * delta) >> 8, 16)
        out[:, i + 2] = sample
    return out.reshape(-1, c).reshape(-1).astype(np.int16)


# --------------------------------------------------------------------------- XACT wave / sound banks

def xwb_entries(head):
    """[(index, codec, channels, rate, block align, data offset, length)] from the start of an .xwb (the bank
    data and entry metadata segments; reading the first 64 KB is enough for a few hundred entries)."""
    if head[:4] != b"WBND":
        raise XnbError("Not an XACT wave bank")
    version = struct.unpack_from("<I", head, 4)[0]
    if version < 42:
        raise XnbError(f"XACT wave bank version {version} isn't supported")
    segs = [struct.unpack_from("<II", head, 12 + 8 * i) for i in range(5)]
    bank = segs[0][0]
    flags, count = struct.unpack_from("<II", head, bank)
    meta_size = struct.unpack_from("<I", head, bank + 72)[0]
    if flags & 0x00020000:
        raise XnbError("Compact wave banks aren't supported")
    data_start = segs[4][0]
    out = []
    for i in range(count):
        at = segs[1][0] + i * meta_size
        _flags_dur, fmt, offset, length = struct.unpack_from("<4I", head, at)
        codec, channels, rate = fmt & 3, (fmt >> 2) & 7, (fmt >> 5) & 0x3FFFF
        align = (fmt >> 23) & 0xFF
        out.append((i, codec, channels, rate, align, data_start + offset, length))
    return out


CODECS = {0: "PCM", 1: "XMA", 2: "MS-ADPCM", 3: "xWMA"}


def xwb_wav(f, entry):
    """WAV bytes of one wave bank entry (PCM or MS-ADPCM); f: the open .xwb file."""
    _i, codec, channels, rate, align, offset, length = entry
    f.seek(offset)
    data = f.read(length)
    if codec == 2:
        block_align = (align + 22) * channels
        return pcm_wav(msadpcm_decode(data, channels, block_align), channels, rate)
    if codec == 0:
        bits = 16  # (8-bit PCM sets the format's top bit; Terraria and FNA games use 16)
        fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
        return wav_file(fmt, data)
    raise XnbError(f"{CODECS.get(codec, codec)} sounds aren't supported")


def xsb_cues(data):
    """{wave index: cue name} for the simple cues of an .xsb sound bank (the names games play music by)."""
    if data[:4] != b"SDBK":
        return {}
    o = 19  # magic, tool / format version, CRC, last modified (8), platform
    n_simple, _n_complex, _u, n_total, _banks, _sounds, names_len, _u2 = struct.unpack_from("<HHHHBHHH", data, o)
    simple, _complex, names_off = struct.unpack_from("<3I", data, o + 15)
    names = data[names_off:names_off + names_len].split(b"\0")
    out: dict[int, str] = {}
    for i in range(min(n_simple, len(names), n_total)):
        try:
            _flags, sound = struct.unpack_from("<BI", data, simple + 5 * i)
            if data[sound] & 1:
                continue  # complex sound: clips and events, no single wave
            track = struct.unpack_from("<H", data, sound + 9)[0]
        except (struct.error, IndexError):
            continue
        out.setdefault(track, names[i].decode("utf-8", "replace"))
    return out


# --------------------------------------------------------------------------- AES (Terraria's player files)


def _aes_tables():
    """AES S-box and its inverse, built from GF(2^8) inverses + the affine map (FIPS-197 5.1.1)."""
    sbox = [0] * 256
    p = q = 1
    while True:  # p walks the multiplicative group by 3, q by 3's inverse
        p ^= ((p << 1) ^ (0x1B if p & 0x80 else 0)) & 0xFF
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ (q << 1 | q >> 7) ^ (q << 2 | q >> 6) ^ (q << 3 | q >> 5) ^ (q << 4 | q >> 4)
        sbox[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    sbox[0] = 0x63
    inv = [0] * 256
    for i, s in enumerate(sbox):
        inv[s] = i
    return sbox, inv


_SBOX, _INV_SBOX = _aes_tables()


def _xtime(a):
    return ((a << 1) ^ 0x1B) & 0xFF if a & 0x80 else a << 1


def _mul(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        a, b = _xtime(a), b >> 1
    return out


_MUL = {m: [_mul(a, m) for a in range(256)] for m in (9, 11, 13, 14)}


def _round_keys(key):
    """AES-128 key schedule: 11 round keys of 16 bytes."""
    words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
    rcon = 1
    for i in range(4, 44):
        t = list(words[i - 1])
        if i % 4 == 0:
            t = [_SBOX[b] for b in t[1:] + t[:1]]
            t[0] ^= rcon
            rcon = _xtime(rcon)
        words.append([a ^ b for a, b in zip(words[i - 4], t)])
    return [sum(words[r * 4:r * 4 + 4], []) for r in range(11)]


def _decrypt_block(block, keys):
    s = [b ^ k for b, k in zip(block, keys[10])]
    m9, m11, m13, m14 = _MUL[9], _MUL[11], _MUL[13], _MUL[14]
    for r in range(9, -1, -1):
        s = [s[(i - 4 * (i % 4)) % 16] for i in range(16)]  # inverse ShiftRows: row i%4 moves right
        s = [_INV_SBOX[b] ^ k for b, k in zip(s, keys[r])]
        if r:
            out = []
            for c in range(0, 16, 4):
                a0, a1, a2, a3 = s[c:c + 4]
                out += [m14[a0] ^ m11[a1] ^ m13[a2] ^ m9[a3], m9[a0] ^ m14[a1] ^ m11[a2] ^ m13[a3],
                        m13[a0] ^ m9[a1] ^ m14[a2] ^ m11[a3], m11[a0] ^ m13[a1] ^ m9[a2] ^ m14[a3]]
            s = out
    return s


def aes_cbc_decrypt(data, key, iv):
    """AES-128-CBC decryption with PKCS#7 padding removed (pure Python: fine for a few KB)."""
    keys = _round_keys(key)
    out, prev = bytearray(), list(iv)
    for i in range(0, len(data) - len(data) % 16, 16):
        block = list(data[i:i + 16])
        out += bytes(a ^ b for a, b in zip(_decrypt_block(block, keys), prev))
        prev = block
    pad = out[-1] if out else 0
    return bytes(out[:-pad] if 1 <= pad <= 16 else out)


PLAYER_KEY = "h3y_gUyZ".encode("utf-16-le")  # Terraria's player file key (also the IV)


# --------------------------------------------------------------------------- .NET metadata (names from Terraria.exe)

# ECMA-335 II.22: the columns of every metadata table, by table number. An int is a fixed size in bytes; "s" / "g" /
# "b" an index into the #Strings / #GUID / #Blob heap; "tN" an index into table N; any other name a coded index.
_TABLE_COLUMNS = [
    "2 s g g g", "Scope s s", "4 s s TypeDefOrRef t4 t6", "t4", "2 s b", "t6", "4 2 2 s b t8", "t8", "2 2 s",
    "t2 TypeDefOrRef", "MemberRefParent s b", "1 1 HasConstant b", "HasCustomAttribute CustomAttributeType b",
    "HasFieldMarshal b", "2 HasDeclSecurity b", "2 4 t2", "4 t4", "b", "t2 t20", "t20", "2 s TypeDefOrRef",
    "t2 t23", "t23", "2 s b", "2 t6 HasSemantics", "t2 MethodDefOrRef MethodDefOrRef", "s", "b",
    "2 MemberForwarded s t26", "4 t4", "4 4", "4", "4 2 2 2 2 4 b s s", "4", "4 4 4", "2 2 2 2 4 b s s b", "4 t35",
    "4 4 4 t35", "4 s b", "4 4 s s Implementation", "4 4 s Implementation", "t2 t2", "2 2 TypeOrMethodDef s",
    "MethodDefOrRef b", "t42 TypeDefOrRef",
]
_CODED = {  # coded index -> (tag bits, the tables it can point to; -1 = unused tag)
    "TypeDefOrRef": (2, (2, 1, 27)), "HasConstant": (2, (4, 8, 23)),
    "HasCustomAttribute": (5, (6, 4, 1, 2, 8, 9, 10, 0, 14, 23, 20, 17, 26, 27, 32, 35, 38, 39, 40, 42, 44, 43)),
    "HasFieldMarshal": (1, (4, 8)), "HasDeclSecurity": (2, (2, 6, 32)), "MemberRefParent": (3, (2, 1, 26, 6, 27)),
    "HasSemantics": (1, (20, 23)), "MethodDefOrRef": (1, (6, 10)), "MemberForwarded": (1, (4, 6)),
    "Implementation": (2, (38, 35, 39)), "CustomAttributeType": (3, (-1, -1, 6, 10, -1)),
    "Scope": (2, (0, 26, 35, 1)), "TypeOrMethodDef": (1, (2, 6)),
}
TYPE_REF, TYPEDEF, FIELD, METHOD_DEF, PARAM, MEMBER_REF, CONSTANT, FIELD_RVA = 1, 2, 4, 6, 8, 10, 11, 29
MANIFEST_RESOURCE, NESTED_CLASS, METHOD_SPEC = 40, 41, 43
# Constant blob formats by ELEMENT_TYPE: bool, char, i1, u1, i2, u2, i4, u4, i8, u8
_CONST_FORMATS = {2: "<?", 3: "<H", 4: "<b", 5: "<B", 6: "<h", 7: "<H", 8: "<i", 9: "<I", 10: "<q", 11: "<Q"}


class DotNetError(ValueError):
    pass


class DotNetAssembly:
    """Just enough of a .NET assembly's metadata (ECMA-335) to get names out of it: the values of `const` fields
    (Terraria's ItemID.IronPickaxe = 1) and embedded resources (its en-US localization JSON).

    PE header -> sections (RVA -> file offset) -> CLI header -> metadata root -> streams (#~ tables, #Strings,
    #Blob) -> the table rows. data: the whole file."""

    def __init__(self, data):
        self.data = data
        self._types = None
        self._methods = None
        try:
            pe = struct.unpack_from("<I", data, 0x3C)[0]
            if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
                raise DotNetError("not a PE file")
            n_sections = struct.unpack_from("<H", data, pe + 6)[0]
            opt_size = struct.unpack_from("<H", data, pe + 20)[0]
            opt = pe + 24
            dirs = opt + (96 if struct.unpack_from("<H", data, opt)[0] == 0x10B else 112)  # PE32 / PE32+
            first = opt + opt_size
            self.sections = [struct.unpack_from("<4I", data, s + 8) for s in range(first, first + 40 * n_sections, 40)]
            cli_rva, cli_size = struct.unpack_from("<II", data, dirs + 14 * 8)
            if not cli_size:
                raise DotNetError("not a .NET assembly")
            cli = self.offset(cli_rva)
            meta_rva, res_rva = struct.unpack_from("<I", data, cli + 8)[0], struct.unpack_from("<I", data, cli + 24)[0]
            self.resources = self.offset(res_rva) if res_rva else 0
            self._read_metadata(self.offset(meta_rva))
        except (struct.error, IndexError, KeyError, ValueError) as e:
            if isinstance(e, DotNetError):
                raise
            raise DotNetError(f"broken .NET metadata: {e!r}") from e

    def offset(self, rva):
        for vsize, va, raw_size, raw in self.sections:
            if va <= rva < va + max(vsize, raw_size):
                return rva - va + raw
        raise DotNetError(f"RVA {rva:#x} is outside the sections")

    def _read_metadata(self, root):
        data = self.data
        if data[root:root + 4] != b"BSJB":
            raise DotNetError("no metadata root")
        o = root + 16 + struct.unpack_from("<I", data, root + 12)[0]  # past the runtime version string
        n_streams = struct.unpack_from("<H", data, o + 2)[0]
        o += 4
        streams = {}
        for _ in range(n_streams):
            off = struct.unpack_from("<I", data, o)[0]
            end = data.index(b"\0", o + 8)
            streams[data[o + 8:end].decode("ascii")] = root + off
            o = root + ((end - root + 4) & ~3)  # names are zero-padded to 4 bytes
        self.strings, self.blob = streams["#Strings"], streams.get("#Blob", 0)
        t = streams["#~"] if "#~" in streams else streams["#-"]
        heap_sizes = data[t + 6]
        valid = struct.unpack_from("<Q", data, t + 8)[0]
        o = t + 24
        self.rows = [0] * 64
        for i in range(64):
            if valid >> i & 1:
                self.rows[i] = struct.unpack_from("<I", data, o)[0]
                o += 4
        if heap_sizes & 0x40:
            o += 4  # extra data in uncompressed (#-) streams
        sizes = {"s": 4 if heap_sizes & 1 else 2, "g": 4 if heap_sizes & 2 else 2, "b": 4 if heap_sizes & 4 else 2}
        for name, (bits, targets) in _CODED.items():
            sizes[name] = 2 if max(self.rows[n] if n >= 0 else 0 for n in targets) < 1 << (16 - bits) else 4
        self.tables = []  # per table number: (file offset, row size, [(column offset, size)])
        for number, columns in enumerate(_TABLE_COLUMNS):
            cols, pos = [], 0
            for c in columns.split():
                if c.isdigit():
                    size = int(c)
                elif c[0] == "t":
                    size = 2 if self.rows[int(c[1:])] < 0x10000 else 4
                else:
                    size = sizes[c]
                cols.append((pos, size))
                pos += size
            self.tables.append((o, pos, cols))
            o += pos * self.rows[number]

    def row(self, table, i):
        """The column values of row i (0-based) of a table."""
        start, size, cols = self.tables[table]
        base = start + size * i
        return [int.from_bytes(self.data[base + p:base + p + n], "little") for p, n in cols]

    def string(self, index):
        start = self.strings + index
        return self.data[start:self.data.index(b"\0", start)].decode("utf-8", "replace")

    def blob_bytes(self, index):
        o = self.blob + index
        b = self.data[o]
        if b < 0x80:
            return self.data[o + 1:o + 1 + b]
        if b < 0xC0:
            return self.data[o + 2:o + 2 + ((b & 0x3F) << 8 | self.data[o + 1])]
        return self.data[o + 4:o + 4 + (int.from_bytes(self.data[o:o + 4], "big") & 0x1FFFFFFF)]

    def type_names(self):
        """Full name of every TypeDef row: "Namespace.Type", nested types as "Namespace.Outer.Inner"."""
        if self._types is not None:
            return self._types
        names = []
        for i in range(self.rows[TYPEDEF]):
            _flags, name, namespace, _extends, _fields, _methods = self.row(TYPEDEF, i)
            names.append((self.string(namespace), self.string(name)))
        outer = {}
        for i in range(self.rows[NESTED_CLASS]):
            nested, enclosing = self.row(NESTED_CLASS, i)
            outer[nested - 1] = enclosing - 1

        def full(i, depth=0):
            if i in outer and depth < 16:
                return f"{full(outer[i], depth + 1)}.{names[i][1]}"
            return f"{names[i][0]}.{names[i][1]}" if names[i][0] else names[i][1]
        self._types = [full(i) for i in range(len(names))]
        return self._types

    def constants(self, type_names):
        """{type name: {field name: value}} of the integer `const` fields of those types (full names, see
        type_names())."""
        n_types, n_fields = self.rows[TYPEDEF], self.rows[FIELD]
        owner = {}  # field row (1-based) -> type name
        for i, full in enumerate(self.type_names()):
            if full in type_names:
                end = self.row(TYPEDEF, i + 1)[4] if i + 1 < n_types else n_fields + 1
                owner.update(dict.fromkeys(range(self.row(TYPEDEF, i)[4], end), full))
        out: dict[str, dict[str, int]] = {name: {} for name in type_names}
        for i in range(self.rows[CONSTANT]):
            etype, _pad, parent, value = self.row(CONSTANT, i)
            field = parent >> 2
            if parent & 3 == 0 and field in owner and etype in _CONST_FORMATS:
                name = self.string(self.row(FIELD, field - 1)[1])
                out[owner[field]][name] = struct.unpack_from(_CONST_FORMATS[etype], self.blob_bytes(value))[0]
        return out

    def methods(self):
        """[(type name, method name, MethodDef row (1-based; token = 0x06000000 | row))] of every method."""
        if self._methods is not None:
            return self._methods
        types, n_types, n_methods = self.type_names(), self.rows[TYPEDEF], self.rows[METHOD_DEF]
        out = []
        for i, type_name in enumerate(types):
            end = self.row(TYPEDEF, i + 1)[5] if i + 1 < n_types else n_methods + 1
            for m in range(self.row(TYPEDEF, i)[5], end):
                out.append((type_name, self.string(self.row(METHOD_DEF, m - 1)[3]), m))
        self._methods = out
        return out

    def method_body(self, method):
        """The IL code of a MethodDef row (1-based), b"" for abstract / extern methods."""
        rva = self.row(METHOD_DEF, method - 1)[0]
        if not rva:
            return b""
        o = self.offset(rva)
        if self.data[o] & 3 == 2:  # tiny header: the size in its top 6 bits
            return self.data[o + 1:o + 1 + (self.data[o] >> 2)]
        return self.data[o + 12:o + 12 + struct.unpack_from("<I", self.data, o + 4)[0]]  # fat header

    def field_token(self, type_name, field_name):
        """Metadata token (0x04000000 | Field row) of a field, or None."""
        types, n_types = self.type_names(), self.rows[TYPEDEF]
        if type_name not in types:
            return None
        i = types.index(type_name)
        end = self.row(TYPEDEF, i + 1)[4] if i + 1 < n_types else self.rows[FIELD] + 1
        for f in range(self.row(TYPEDEF, i)[4], end):
            if self.string(self.row(FIELD, f - 1)[1]) == field_name:
                return 0x04000000 | f
        return None

    def param_names(self, token):
        """Parameter names of a MethodDef ([] for other tokens). Param rows: flags, sequence (0 = return), name."""
        if token >> 24 != METHOD_DEF:
            return []
        row = (token & 0xFFFFFF) - 1
        first = self.row(METHOD_DEF, row)[5]
        end = self.row(METHOD_DEF, row + 1)[5] if row + 1 < self.rows[METHOD_DEF] else self.rows[PARAM] + 1
        params = [self.row(PARAM, p - 1) for p in range(first, end)]
        return [self.string(name) for _flags, sequence, name in params if sequence > 0]

    def _method_token(self, token):
        """A MethodSpec (generic instantiation) token -> its MethodDef / MemberRef token."""
        if token >> 24 == 0x2B:
            coded = self.row(METHOD_SPEC, (token & 0xFFFFFF) - 1)[0]
            return (0x0A000000 if coded & 1 else 0x06000000) | coded >> 1
        return token

    def member_name(self, token):
        """Name of a Field / MethodDef / MemberRef / MethodSpec / TypeRef / TypeDef token."""
        token = self._method_token(token)
        table, row = token >> 24, (token & 0xFFFFFF) - 1
        column = {FIELD: 1, METHOD_DEF: 3, MEMBER_REF: 1, TYPE_REF: 1, TYPEDEF: 1}.get(table)
        return self.string(self.row(table, row)[column]) if column is not None and row >= 0 else ""

    def signature(self, token):
        """(has `this`, parameter count, returns void) of a method token, or None. Signature blob: flags (0x20
        = has this, 0x10 = generic: a generic parameter count follows), parameter count, return type..."""
        token = self._method_token(token)
        table, row = token >> 24, (token & 0xFFFFFF) - 1
        if table not in (METHOD_DEF, MEMBER_REF):
            return None
        sig = self.blob_bytes(self.row(table, row)[4] if table == METHOD_DEF else self.row(table, row)[2])
        i = 2 if sig[0] & 0x10 else 1
        return bool(sig[0] & 0x20), sig[i], sig[i + 1] == 0x01  # ELEMENT_TYPE_VOID

    def field_data(self, token, size):
        """`size` bytes of a field's initial data (FieldRVA: what `new int[] {1, 2, ...}` arrays are copied from),
        or None."""
        for i in range(self.rows[FIELD_RVA]):
            rva, field = self.row(FIELD_RVA, i)
            if field == token & 0xFFFFFF:
                o = self.offset(rva)
                return self.data[o:o + size]
        return None

    def resource_names(self):
        return [self.string(self.row(MANIFEST_RESOURCE, i)[2]) for i in range(self.rows[MANIFEST_RESOURCE])]

    def resource(self, name):
        """The bytes of an embedded manifest resource, or None."""
        for i in range(self.rows[MANIFEST_RESOURCE]):
            offset, _flags, res_name, implementation = self.row(MANIFEST_RESOURCE, i)
            if implementation == 0 and self.string(res_name) == name:
                o = self.resources + offset
                return self.data[o + 4:o + 4 + struct.unpack_from("<I", self.data, o)[0]]
        return None


# CIL operand sizes (ECMA-335 III): one-byte opcodes, then the 0xFE-prefixed ones. switch (0x45) is variable.
_OPERANDS = bytearray(256)
for _ops, _n in ((range(0x0E, 0x14), 1), ((0x1F, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35,
                                           0x36, 0x37, 0xDE), 1),
                 ((0x20, 0x22, 0x27, 0x28, 0x29, 0x79, 0x8C, 0x8D, 0x8F, 0xA3, 0xA4, 0xA5, 0xC2, 0xC6, 0xD0, 0xDD), 4),
                 (range(0x38, 0x45), 4), (range(0x6F, 0x76), 4), (range(0x7B, 0x82), 4), ((0x21, 0x23), 8)):
    for _op in _ops:
        _OPERANDS[_op] = _n
_OPERANDS_FE = {0x06: 4, 0x07: 4, 0x09: 2, 0x0A: 2, 0x0B: 2, 0x0C: 2, 0x0D: 2, 0x0E: 2, 0x12: 1, 0x15: 4, 0x16: 4,
                0x19: 1, 0x1C: 4}
DUP, POP, CALL, NEWOBJ, STFLD, LDSFLD, STSFLD, NEWARR = 0x25, 0x26, 0x28, 0x73, 0x7D, 0x7E, 0x80, 0x8D
LDELEM_REF, STELEM_I4, STELEM_REF, LDELEM, STELEM, LDTOKEN = 0x9A, 0x9E, 0xA2, 0xA3, 0xA4, 0xD0
CALLVIRT, LDC_R4, LDC_R8, SWITCH = 0x6F, 0x22, 0x23, 0x45
_LDLOC = {0x06, 0x07, 0x08, 0x09, 0x11, 0xFE0C}  # ldloc.0-3, ldloc.s, ldloc
_STLOC = {0x0A, 0x0B, 0x0C, 0x0D, 0x13, 0xFE0E}
_LDLOCA = {0x12, 0xFE0D}


def _local_index(op, arg):
    """The local variable a ldloc / stloc / ldloca instruction uses."""
    if 0x06 <= op <= 0x09:
        return op - 0x06
    if 0x0A <= op <= 0x0D:
        return op - 0x0A
    return arg[0] if len(arg) == 1 else struct.unpack("<H", arg)[0]


def il_walk(code):
    """[(offset, opcode, operand bytes)] of a method's IL (0xFE-prefixed opcodes as 0xFE00 | second byte)."""
    out, i = [], 0
    while i < len(code):
        op = code[i]
        if op == 0xFE and i + 1 < len(code):
            n = _OPERANDS_FE.get(code[i + 1], 0)
            out.append((i, 0xFE00 | code[i + 1], code[i + 2:i + 2 + n]))
            i += 2 + n
        elif op == SWITCH:  # count, then that many branch offsets (the operand keeps them all)
            n = 4 + 4 * struct.unpack_from("<I", code, i + 1)[0]
            out.append((i, op, code[i + 1:i + 1 + n]))
            i += 1 + n
        else:
            n = _OPERANDS[op]
            out.append((i, op, code[i + 1:i + 1 + n]))
            i += 1 + n
    return out


def il_instructions(code):
    """[(opcode, operand bytes)] of a method's IL."""
    return [(op, arg) for _offset, op, arg in il_walk(code)]


def il_branch_target(offset, op, arg):
    """Where a branch instruction at `offset` jumps to (short forms 0x2B-0x37, long 0x38-0x44), or None."""
    if 0x2B <= op <= 0x37:
        return offset + 2 + struct.unpack("<b", arg)[0]
    if 0x38 <= op <= 0x44:
        return offset + 5 + struct.unpack("<i", arg)[0]
    return None


def il_int(op, arg):
    """The value an ldc.i4 instruction pushes, or None for any other instruction."""
    if 0x15 <= op <= 0x1E:  # ldc.i4.m1, ldc.i4.0 ... ldc.i4.8
        return op - 0x16
    if op == 0x1F:
        return struct.unpack("<b", arg)[0]
    if op == 0x20:
        return struct.unpack("<i", arg)[0]
    return None


def il_token(arg):
    return struct.unpack("<I", arg)[0]


LDARG_0, LDARG_1, RET, BR_S, BR, SUB, LDFLD = 0x02, 0x03, 0x2A, 0x2B, 0x38, 0x59, 0x7B
_BEQ, _BNE = (0x2E, 0x3B), (0x33, 0x40)
# Item helpers whose arguments are field values: name -> the fields, in parameter order.
IL_HELPERS = {"SetShopValues": ("rare", "value"), "SetWeaponValues": ("damage", "knockBack", "crit"),
              "DefaultToPlaceableTile": ("createTile", "placeStyle"), "DefaultToPlaceableWall": ("createWall",),
              "DefaultToAccessory": ("width", "height")}


def _il_value(asm, ins, i):
    """(value, index after it) of the constant expression starting at instruction i: an int / float constant,
    or Item.buyPrice / sellPrice(platinum, gold, silver, copper) (value in copper; selling pays a fifth).
    (None, None) if it is anything else."""
    if i >= len(ins):
        return None, None
    _offset, op, arg = ins[i]
    value = il_int(op, arg)
    if value is not None:
        if i + 4 < len(ins) and ins[i + 4][1] == CALL:
            name = asm.member_name(il_token(ins[i + 4][2]))
            coins = [il_int(o, a) for _off, o, a in ins[i:i + 4]]
            if name in ("buyPrice", "sellPrice") and None not in coins:
                p, g, s, c = coins
                return (p * 1_000_000 + g * 10_000 + s * 100 + c) * (5 if name == "sellPrice" else 1), i + 5
        return value, i + 1
    if op == LDC_R4:
        return round(struct.unpack("<f", arg)[0], 4), i + 1
    if op == LDC_R8:
        return struct.unpack("<d", arg)[0], i + 1
    return None, None


def il_defaults(asm, method, type_field, out):
    """Fill out {type: {field: value}} from the `if (type == N) { this.damage = 12; ... }` chains of a
    SetDefaults method (items, NPCs, projectiles). A condition is `ldarg.1` (the type parameter) or
    `this.type`, then ldc N and bne.un (skip the block unless equal); `type == a || type == b` adds beq jumps to
    the block. `switch (type - base)` jump tables work too (a case runs to its ret / br). Inside a block:
    `ldarg.0, <constant>, stfld field`, and a few helpers (SetShopValues(rare, value)...)."""
    ins = il_walk(asm.method_body(method))
    current: set = set()
    pending: dict[int, set] = {}  # block start -> types whose `type == N` jumps there (beq)
    cases: dict[int, set] = {}  # switch case start -> types
    end, k = -1, 0
    in_case = False
    while k < len(ins):
        offset, op, arg = ins[k]
        if offset in cases:
            current, end, in_case = cases[offset], 1 << 62, True
        elif offset >= end or in_case and op in (RET, BR, BR_S):
            current, in_case = set(), False
        if op == SWITCH and k >= 3 and ins[k - 1][1] == SUB and il_int(*ins[k - 2][1:]) is not None                 and ins[k - 3][1] in (LDARG_1, LDFLD):
            base, n = il_int(*ins[k - 2][1:]), struct.unpack_from("<I", arg)[0]
            after = offset + 1 + len(arg)
            for i, rel in enumerate(struct.unpack_from(f"<{n}i", arg, 4)):
                cases.setdefault(after + rel, set()).add(base + i)
            k += 1
            continue
        j = None
        if op == LDARG_1:
            j = k + 1
        elif op == LDARG_0 and k + 1 < len(ins) and ins[k + 1][1] == LDFLD and il_token(ins[k + 1][2]) == type_field:
            j = k + 2
        if j is not None and j + 1 < len(ins):
            number = il_int(*ins[j][1:])
            branch = ins[j + 1]
            if number is not None and branch[1] in _BEQ + _BNE:
                target = il_branch_target(*branch)
                if branch[1] in _BEQ:
                    pending.setdefault(target, set()).add(number)
                else:
                    start = ins[j + 2][0] if j + 2 < len(ins) else -1
                    current, end = {number} | pending.pop(start, set()), target
                k = j + 2
                continue
        if current and op == LDARG_0:
            value, m = _il_value(asm, ins, k + 1)
            if m is not None and m < len(ins) and ins[m][1] == STFLD:
                field = asm.member_name(il_token(ins[m][2]))
                for number in current:
                    out.setdefault(number, {})[field] = value
                k = m + 1
                continue
            args, m = [], k + 1  # this.Helper(constants...)
            while True:
                value, after = _il_value(asm, ins, m)
                if after is None:
                    break
                args.append(value)
                m = after
            if args and m < len(ins) and ins[m][1] in (CALL, CALLVIRT):
                name = asm.member_name(il_token(ins[m][2]))
                fields = IL_HELPERS.get(name, ())
                if not fields:  # show the call: DefaultToWhip(projectileId=913, dmg=12, ...)
                    params = asm.param_names(il_token(ins[m][2]))
                    call = f"{name}(" + ", ".join(f"{p}={v}" if p else str(v) for p, v in
                                                  zip(params + [""] * len(args), args)) + ")"
                for number in current:
                    if not fields:
                        out.setdefault(number, {}).setdefault("calls", []).append(call)
                        continue
                    out.setdefault(number, {}).update(zip(fields, args))
                    if fields and fields[0] == "width" and len(fields) == 2:
                        out[number]["accessory"] = 1
                k = m + 1
                continue
        k += 1
    return out


class IlArray(dict):
    """An array made by `newarr` while evaluating IL: {index: value}, with its length and element type name."""

    def __init__(self, size=None, element=""):
        super().__init__()
        self.size, self.element = size, element


_ELEMENT_FORMATS = {"Int32": "i", "UInt32": "I", "Int16": "h", "UInt16": "H", "Byte": "B", "SByte": "b",
                    "Boolean": "?", "Single": "f", "Int64": "q", "Double": "d"}
_STELEM = {0x9B, 0x9C, 0x9D, 0x9E, 0x9F, 0xA0, 0xA1, 0xA2, 0xA4}  # stelem.i, .i1, .i2, .i4, .i8, .r4, .r8, .ref, <T>


def il_evaluate(asm, method, on_call=None, on_store=None):
    """Run a method's IL straight through on a tiny stack machine - enough for the data-filling code Terraria's
    static constructors and Initialize methods are made of: constants, locals, static fields, arrays (stelem,
    RuntimeHelpers.InitializeArray) and calls. Branches are ignored and any instruction it doesn't know empties
    the stack, so it only sees straight-line runs like `array[5] = new Color(1, 2, 3)`.

    on_call(name, args) -> value: for call / callvirt / newobj (`args` without `this`; newobj passes ".ctor" with
    the type's token as name prefix: "<token>.ctor"); return None if unknown.
    on_store(field token, value): for stsfld.
    Returns the locals {index: value}."""
    local: dict[int, object] = {}
    stack: list = []
    for op, arg in il_instructions(asm.method_body(method)):
        value = il_int(op, arg)
        try:
            if value is not None:
                stack.append(value)
            elif op == LDC_R4:
                stack.append(struct.unpack("<f", arg)[0])
            elif op == LDC_R8:
                stack.append(struct.unpack("<d", arg)[0])
            elif op in _LDLOC:
                stack.append(local.get(_local_index(op, arg)))
            elif op in _STLOC:
                local[_local_index(op, arg)] = stack.pop()
            elif op in _LDLOCA:
                stack.append(("address", _local_index(op, arg)))
            elif op == LDSFLD:
                stack.append(("field", il_token(arg)))
            elif op == STSFLD:
                item = stack.pop()
                if on_store:
                    on_store(il_token(arg), item)
            elif op == LDTOKEN:
                stack.append(("token", il_token(arg)))
            elif op == NEWARR:
                stack.append(IlArray(stack.pop(), asm.member_name(il_token(arg))))
            elif op in (CALL, CALLVIRT, NEWOBJ):
                token = il_token(arg)
                has_this, n, returns_void = asm.signature(token) or (False, 0, True)
                args = [stack.pop() for _ in range(n)][::-1]
                name = asm.member_name(token)
                this = stack.pop() if has_this and op != NEWOBJ else None
                if name == "InitializeArray" and len(args) == 2 and isinstance(args[0], IlArray):
                    array, (_tag, field) = args
                    fmt = _ELEMENT_FORMATS.get(array.element)
                    if fmt and isinstance(array.size, int):
                        data = asm.field_data(field, struct.calcsize(fmt) * array.size)
                        if data:
                            array.update(enumerate(struct.unpack(f"<{array.size}{fmt}", data)))
                    continue
                result = on_call(name, args) if on_call else None
                if op == CALL and name == ".ctor" and isinstance(this, tuple) and this[0] == "address":
                    local[this[1]] = result  # `Color c = new Color(...)` built in place on a local
                elif op == NEWOBJ or not returns_void:
                    stack.append(result)
            elif op == DUP:
                stack.append(stack[-1])
            elif op == LDELEM_REF:
                index, array = stack.pop(), stack.pop()
                stack.append(array.setdefault(index, IlArray()) if isinstance(array, IlArray) else None)
            elif op == LDELEM:
                index, array = stack.pop(), stack.pop()
                stack.append(array.get(index) if isinstance(array, IlArray) else None)
            elif op in _STELEM:
                item, index, array = stack.pop(), stack.pop(), stack.pop()
                if isinstance(array, IlArray) and isinstance(index, int):
                    array[index] = item
            elif op == POP:
                stack.pop()
            else:
                stack.clear()
        except IndexError:
            stack.clear()
    return local



# --------------------------------------------------------------------------- Terraria names

# File prefix ("Item" of Images/Item_1.xnb) -> (label, ID class in Terraria.ID, localization section first tried).
# Images/Armor/Armor_N are the body sheets. Names not in the localization (tiles, walls, gores...) are made from
# the internal name: "DirtUnsafe" -> "Dirt Unsafe".
NAME_PREFIXES = {
    "Item": ("Item", "ItemID", "ItemName"), "NPC": ("NPC", "NPCID", "NPCName"),
    "NPC_Head": ("NPC head", "NPCHeadID", "NPCName"), "Projectile": ("Projectile", "ProjectileID", "ProjectileName"),
    "Buff": ("Buff", "BuffID", "BuffName"), "Tiles": ("Tile", "TileID", ""), "Wall": ("Wall", "WallID", ""),
    "Gore": ("Gore", "GoreID", ""), "Extra": ("Extra", "ExtrasID", ""), "Glow": ("Glow", "GlowMaskID", "ItemName"),
    "Armor_Head": ("Armor head", "ArmorIDs.Head", "ItemName"), "Armor": ("Armor body", "ArmorIDs.Body", "ItemName"),
    "Armor_Legs": ("Armor legs", "ArmorIDs.Legs", "ItemName"), "Wings": ("Wings", "ArmorIDs.Wing", "ItemName"),
    "Acc_Back": ("Back", "ArmorIDs.Back", "ItemName"), "Acc_Balloon": ("Balloon", "ArmorIDs.Balloon", "ItemName"),
    "Acc_Beard": ("Beard", "ArmorIDs.Beard", "ItemName"), "Acc_Face": ("Face", "ArmorIDs.Face", "ItemName"),
    "Acc_Front": ("Front", "ArmorIDs.Front", "ItemName"), "Acc_HandsOff": ("Hands off", "ArmorIDs.HandOff", "ItemName"),
    "Acc_HandsOn": ("Hands on", "ArmorIDs.HandOn", "ItemName"), "Acc_Neck": ("Neck", "ArmorIDs.Neck", "ItemName"),
    "Acc_Shield": ("Shield", "ArmorIDs.Shield", "ItemName"), "Acc_Shoes": ("Shoes", "ArmorIDs.Shoe", "ItemName"),
    "Acc_Waist": ("Waist", "ArmorIDs.Waist", "ItemName"), "Music": ("Music", "MusicID", ""),
}
# Main's per-tile / per-wall flag arrays worth showing (filled with `Main.tileSolid[1] = true` in Main's methods).
MAIN_TILE_ARRAYS = ("tileSolid", "tileSolidTop", "tileFrameImportant", "tileLighted", "tileTable", "tileContainer",
                    "tileNoAttach", "tileLavaDeath", "tileCut", "tileAxe", "tileHammer", "tileDungeon", "tileShine",
                    "tileShine2", "tileSpelunker", "tileBrick", "tileStone", "tileMergeDirt", "tileBlockLight",
                    "tileWaterDeath", "tileNoFail", "tileObsidianKill", "tileSand", "tileFlame", "tileRope",
                    "tileOreFinderPriority", "wallHouse", "wallLight", "wallDungeon", "wallBlend", "wallLargeFrames")
LOCALIZATION = ("Items", "NPCs", "Projectiles", "Game")  # Terraria.Localization.Content.en-US.<name>.json
_NUMBERED = re.compile(r"^(.+)_(\d+)$")


def spaced(name):
    """'OverworldDay' -> 'Overworld Day', 'Boss1' -> 'Boss 1'."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)", " ", name)


def lenient_json(data):
    """Terraria's JSON (read by Newtonsoft) has trailing commas."""
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", data.decode("utf-8-sig")))


class TerrariaData:
    """id -> (internal name, display name) per ID class, from the game's own exe: the `const` fields of
    Terraria.ID.ItemID etc. and the embedded en-US localization."""

    def __init__(self, exe_data):
        asm = DotNetAssembly(exe_data)
        classes = {f"Terraria.ID.{cls}" for _label, cls, _section in NAME_PREFIXES.values()} | {"Terraria.ID.PrefixID"}
        self.ids: dict[str, dict[int, str]] = {}
        for full, fields in asm.constants(classes).items():
            by_id: dict[int, str] = {}
            for field, value in fields.items():
                if field != "Count":  # ItemID.Count etc.: the number of IDs
                    by_id.setdefault(value, field)
            self.ids[full[len("Terraria.ID."):]] = by_id
        self.text: dict[str, dict[str, str]] = {}  # section -> {key: text}
        for file in LOCALIZATION:
            data = asm.resource(f"Terraria.Localization.Content.en-US.{file}.json")
            if data is None:
                continue
            try:
                sections = lenient_json(data)
            except ValueError as e:
                log.warning("Terraria localization %s: %s", file, e)
                continue
            for section, entries in sections.items():
                if isinstance(entries, dict):
                    self.text.setdefault(section, {}).update((k, v) for k, v in entries.items() if isinstance(v, str))
        self.npc_frames: dict[int, int] = {}  # NPC type -> frames in its sheet (Main.npcFrameCount)
        self.projectile_frames: dict[int, int] = {}  # Main.projFrames
        self.item_animations: dict[int, tuple[int, int, bool]] = {}  # item -> (ticks per frame, frames, ping-pong)
        try:
            self._read_frames(asm)
        except (DotNetError, struct.error, IndexError) as e:
            log.warning("Terraria animation frames: %s", e)
        self.tile_colors: dict[int, tuple] = {}  # tile type -> map color (r, g, b) (MapHelper.Initialize)
        self.wall_colors: dict[int, tuple] = {}
        try:
            self._read_map_colors(asm)
        except (DotNetError, struct.error, IndexError, ValueError) as e:
            log.warning("Terraria map colors: %s", e)
        self.defaults: dict[str, dict[int, dict]] = {}  # "Item" / "NPC" / "Projectile" -> type -> {field: value}
        self.sets: dict[str, dict[str, dict[int, object]]] = {}  # "Item" / "NPC" / "Tile"... -> set -> {id: value}
        self.main_arrays: dict[str, dict[int, object]] = {}  # Main.tileSolid etc. -> {id: value}
        for what, reader in (("stats", self._read_defaults), ("ID sets", self._read_sets),
                             ("tile flags", self._read_main_arrays)):
            try:
                reader(asm)
            except (DotNetError, struct.error, IndexError, ValueError, KeyError) as e:
                log.warning("Terraria %s: %s", what, e)

    def _read_defaults(self, asm):
        """Item / NPC / projectile stats from their SetDefaults methods (see il_defaults)."""
        for cls, label in (("Terraria.Item", "Item"), ("Terraria.NPC", "NPC"), ("Terraria.Projectile", "Projectile")):
            type_field = asm.field_token(cls, "type")
            out = self.defaults.setdefault(label, {})
            for type_name, name, row in asm.methods():
                if type_name == cls and name.startswith("SetDefaults") and name != "SetDefaultsFromNetId":
                    il_defaults(asm, row, type_field, out)

    def _read_sets(self, asm):
        """ItemID.Sets, NPCID.Sets, TileID.Sets...: static fields made by `Factory.CreateBoolSet(default, ids...)`
        (the listed ids get the other value) and CreateIntSet / CreateUshortSet / CreateFloatSet(default, id,
        value, id, value...). Only the entries that differ from the default are kept."""
        for label, cls in (("Item", "ItemID"), ("NPC", "NPCID"), ("Projectile", "ProjectileID"), ("Tile", "TileID"),
                           ("Wall", "WallID"), ("Buff", "BuffID")):
            method = next((row for type_name, name, row in asm.methods()
                           if type_name == f"Terraria.ID.{cls}.Sets" and name == ".cctor"), None)
            if method is None:
                continue
            found: dict[str, dict[int, object]] = {}

            def make(name, args):
                if not name.startswith("Create") or not name.endswith("Set") or name == "CreateCustomSet":
                    return None
                if not args or not isinstance(args[-1], IlArray):
                    return None
                values = [args[-1].get(i) for i in range(args[-1].size if isinstance(args[-1].size, int) else 0)]
                if name == "CreateBoolSet":
                    off = not (len(args) == 2 and args[0])  # the listed ids get the opposite of the default
                    return ("set", {i: off for i in values if isinstance(i, int)})
                default = args[0] if len(args) == 2 else 0
                pairs = zip(values[::2], values[1::2])
                return ("set", {int(i): v for i, v in pairs if isinstance(i, (int, float)) and v != default})

            def store(token, value, found=found):
                if isinstance(value, tuple) and value and value[0] == "set" and value[1]:
                    found[asm.member_name(token)] = value[1]

            il_evaluate(asm, method, on_call=make, on_store=store)
            self.sets[label] = found

    def _read_main_arrays(self, asm):
        """Main.tileSolid[5] = true; Main.tileLighted[4] = true... (ldsfld, ldc, ldc, stelem) in Main's methods:
        the per-tile / per-wall flag arrays."""
        main = {asm.field_token("Terraria.Main", n): n for n in MAIN_TILE_ARRAYS}
        main.pop(None, None)
        for type_name, _name, row in asm.methods():
            if type_name != "Terraria.Main":
                continue
            ins = il_instructions(asm.method_body(row))
            for k in range(len(ins) - 3):
                op, arg = ins[k]
                if op == LDSFLD and il_token(arg) in main and ins[k + 3][0] in _STELEM:
                    index, value = il_int(*ins[k + 1]), il_int(*ins[k + 2])
                    if index is not None and value is not None:
                        self.main_arrays.setdefault(main[il_token(arg)], {})[index] = value

    def flags(self, label, number):
        """[(set name, value)] of the ID sets an item / NPC / tile... is in (bool sets as value True)."""
        return [(name, values[number]) for name, values in sorted(self.sets.get(label, {}).items())
                if number in values]

    def item_name(self, item):
        found = self.lookup("Item", item)
        return found[2] if found else f"item #{item}"

    def prefix_name(self, prefix):
        internal = self.ids.get("PrefixID", {}).get(prefix)
        return (self.display("Prefix", internal) or spaced(internal)) if internal else ""

    def _read_map_colors(self, asm):
        """The map colors of tiles and walls. MapHelper.Initialize fills `Color[][] tileColors = new
        Color[TileID.Count][]` and one for walls with `tileColors[type][option] = new Color(r, g, b)` (or a Color
        local made earlier, or another entry); the first option's color is used."""
        method = next((row for type_name, name, row in asm.methods()
                       if type_name == "Terraria.Map.MapHelper" and name == "Initialize"), None)
        if method is None:
            return
        counts = {asm.field_token("Terraria.ID.TileID", "Count"): self.tile_colors,
                  asm.field_token("Terraria.ID.WallID", "Count"): self.wall_colors}

        def color(name, args):
            if name == ".ctor" and len(args) in (3, 4) and all(isinstance(a, int) for a in args):
                return tuple(args[:3])
            return None

        for value in il_evaluate(asm, method, on_call=color).values():
            if isinstance(value, IlArray) and isinstance(value.size, tuple) and value.size[1] in counts:
                target = counts[value.size[1]]
                for kind, options in value.items():
                    if isinstance(kind, int) and isinstance(options, IlArray):
                        first = [options[o] for o in sorted(o for o in options if isinstance(o, int))
                                 if isinstance(options[o], tuple)]
                        if first:
                            target[kind] = first[0]

    def _read_frames(self, asm):
        """Frame counts straight from the game's IL:
            Main.npcFrameCount = new int[] {1, 2, 2, ...}            (Main..cctor: a FieldRVA array)
            Main.projFrames[type] = n;  Main.npcFrameCount[type] = n  (ldsfld, ldc, ldc, stelem.i4)
            Main.RegisterItemAnimation(type, new DrawAnimationVertical(ticks, frames[, pingPong]))"""
        arrays = {asm.field_token("Terraria.Main", "npcFrameCount"): self.npc_frames,
                  asm.field_token("Terraria.Main", "projFrames"): self.projectile_frames}
        arrays.pop(None, None)
        methods = asm.methods()
        owners = {row: type_name for type_name, _name, row in methods}
        register = next((0x06000000 | row for type_name, name, row in methods
                         if type_name == "Terraria.Main" and name == "RegisterItemAnimation"), None)
        needles = [struct.pack("<I", t) for t in list(arrays) + [register] if t]
        literal, sets = {}, []
        for _type_name, _name, row in methods:
            code = asm.method_body(row)
            if not any(n in code for n in needles):
                continue
            ins = il_instructions(code)
            for k, (op, arg) in enumerate(ins):
                if op == STSFLD and il_token(arg) in arrays and k >= 5 and ins[k - 2][0] == LDTOKEN:
                    count = il_int(*ins[k - 5])
                    data = asm.field_data(il_token(ins[k - 2][1]), 4 * count) if count else None
                    if data and len(data) == 4 * count:
                        literal[il_token(arg)] = struct.unpack(f"<{count}i", data)
                elif op == LDSFLD and il_token(arg) in arrays and k + 3 < len(ins) and ins[k + 3][0] == STELEM_I4:
                    index, value = il_int(*ins[k + 1]), il_int(*ins[k + 2])
                    if index is not None and value is not None:
                        sets.append((il_token(arg), index, value))
                elif op == CALL and register and il_token(arg) == register:
                    j, fields = k - 1, {}  # `new DrawAnimationVertical(...) { PingPong = true }`: dup, ldc, stfld
                    while j >= 3 and ins[j][0] == STFLD and ins[j - 2][0] == DUP and il_token(ins[j][1]) >> 24 == 4:
                        fields[asm.string(asm.row(FIELD, (il_token(ins[j][1]) & 0xFFFFFF) - 1)[1])] = il_int(*ins[j - 1])
                        j -= 3
                    ctor = il_token(ins[j][1]) if ins[j][0] == NEWOBJ else 0
                    if ctor >> 24 != 6 or not owners.get(ctor & 0xFFFFFF, "").endswith("DrawAnimationVertical"):
                        continue
                    n_args = asm.signature(ctor)[1]
                    values = [il_int(*ins[i]) for i in range(j - 1 - n_args, j)]  # item type, ctor args
                    if len(values) >= 3 and None not in values and not fields.get("NotActuallyAnimating"):
                        item, ticks, frames, *rest = values
                        self.item_animations[item] = (ticks, frames, bool(fields.get("PingPong") or rest and rest[0]))
        for token, numbers in literal.items():
            arrays[token].update(enumerate(numbers))
        for token, index, value in sets:
            arrays[token][index] = value

    def display(self, section, key):
        """Localized text, following "{$ItemName.Key}" references; "" if there is none."""
        for _ in range(4):
            text = self.text.get(section, {}).get(key, "")
            m = re.fullmatch(r"\{\$(\w+)\.(\w+)\}", text)
            if not m:
                return text
            section, key = m.groups()
        return ""

    def lookup(self, prefix, number):
        """(label, internal name, display name) for a file prefix and number (Item, 1), or None."""
        rule = NAME_PREFIXES.get(prefix)
        if rule is None:
            return None
        label, cls, section = rule
        internal = self.ids.get(cls, {}).get(number)
        if not internal:
            return None
        shown = (section and self.display(section, internal)) or self.display("ItemName", internal) or spaced(internal)
        return label, internal, shown


NPC_SECONDS, PROJECTILE_SECONDS = 6 / 60, 5 / 60  # typical frame times (the game's AI code picks per type)


def animation_of(game, stem):
    """(frames, seconds per frame, ping-pong) of an animated sheet (file stem "NPC_4"), or None."""
    m = _NUMBERED.match(stem)
    if not m:
        return None
    prefix, number = m.group(1), int(m.group(2))
    if prefix == "Item" and number in game.item_animations:
        ticks, frames, ping_pong = game.item_animations[number]
        return (frames, ticks / 60, ping_pong) if frames > 1 else None
    frames = {"NPC": game.npc_frames, "Projectile": game.projectile_frames}.get(prefix, {}).get(number, 1)
    return (frames, NPC_SECONDS if prefix == "NPC" else PROJECTILE_SECONDS, False) if frames > 1 else None


def _grid(w, h, cell_w, cell_h, step_x, step_y):
    return [(x, y, min(cell_w, w - x), min(cell_h, h - y))
            for y in range(0, h - cell_h // 2, step_y) for x in range(0, w - cell_w // 2, step_x)]


def sheet_frames(game, stem, w, h):
    """The frames of a Terraria sprite sheet [(x, y, w, h)], y from the top; [] if it isn't one.
    Animated NPC / projectile / item sheets are vertical strips (frame height = height / frame count). Player
    parts, armor heads / legs, hair and most accessories: 40 px wide frames every 56 px; body sheets (360x224)
    a 9x4 grid of them. Tiles: 16 px cells every 18 px; walls: 32 px every 36 px."""
    anim = animation_of(game, stem)
    if anim:
        step = h // anim[0]
        return [(0, i * step, w, step) for i in range(anim[0])]
    m = _NUMBERED.match(stem)
    prefix = m.group(1) if m else stem
    if (w, h) == (360, 224):
        return _grid(w, h, 40, 56, 40, 56)
    if w == 40 and h >= 112 and h % 56 in (0, 54):
        return _grid(w, h, 40, 56, 40, 56)
    if prefix == "Tiles" and w % 18 in (0, 16) and h % 18 in (0, 16) and w > 18:
        return _grid(w, h, 16, 16, 18, 18)
    if prefix == "Wall" and w % 36 in (0, 32) and h % 36 in (0, 32) and w > 36:
        return _grid(w, h, 32, 32, 36, 36)
    return []


def frame_order(frames, ping_pong):
    """Frame indexes to play: 0..n-1, and back down for ping-pong animations."""
    return list(range(frames)) + (list(range(frames - 2, 0, -1)) if ping_pong else [])


def gif_bytes(images, seconds):
    """An endless-loop animated GIF of RGBA PIL images."""
    out = io.BytesIO()
    images[0].save(out, "GIF", save_all=True, append_images=images[1:], duration=max(20, round(seconds * 1000)),
                   loop=0, disposal=2)
    return out.getvalue()


RARITIES = {-13: "Master (fiery red)", -12: "Expert (rainbow)", -11: "Quest (amber)", -1: "Gray", 0: "White",
            1: "Blue", 2: "Green", 3: "Orange", 4: "Light red", 5: "Pink", 6: "Light purple", 7: "Lime", 8: "Yellow",
            9: "Cyan", 10: "Red", 11: "Purple"}
DAMAGE_CLASSES = ("melee", "ranged", "magic", "summon", "thrown")
# Fields not worth a row of their own (drawing / animation details); they still show under "Other".
_QUIET_FIELDS = {"width", "height", "useStyle", "useTurn", "alpha", "scale", "noUseGraphic", "noMelee", "UseSound",
                 "holdStyle", "placeStyle", "shootsEveryUse", "useAnimation"}


def coins(copper):
    """1234567 -> '1 platinum 23 gold 45 silver 67 copper'."""
    copper = int(copper)
    parts = [(copper // 1_000_000, "platinum"), (copper // 10_000 % 100, "gold"), (copper // 100 % 100, "silver"),
             (copper % 100, "copper")]
    return " ".join(f"{n} {name}" for n, name in parts if n) or "none"


def game_rows(game, label, number):
    """Info panel rows of what the game's code says about an item / NPC / projectile / tile / wall / buff: stats
    from SetDefaults, ID set flags, Main's tile flags, map color."""
    data_label = {"Tile": "Tile", "Wall": "Wall", "Buff": "Buff", "Item": "Item", "NPC": "NPC",
                  "Projectile": "Projectile"}.get(label)
    if game is None or data_label is None:
        return []
    d = dict(game.defaults.get(data_label, {}).get(number, {}))
    rows, used = [], set()

    def name_of(prefix, n):
        found = game.lookup(prefix, n)
        return f"{found[2]} ({n})" if found else str(n)

    def row(title, field, fmt=str):
        if d.get(field) not in (None, 0, 0.0):
            rows.append((title, fmt(d[field])))
        used.add(field)

    if data_label == "Item":
        kind = [c for c in DAMAGE_CLASSES if d.get(c)]
        used.update(DAMAGE_CLASSES)
        row("Damage", "damage", lambda v: f"{v}" + (f" ({', '.join(kind)})" if kind else ""))
        row("Critical chance", "crit", lambda v: f"+{v}%")
        row("Knockback", "knockBack")
        row("Use time", "useTime", lambda v: f"{v} ticks ({60 / v:.1f} per second)" if v > 0 else str(v))
        row("Mana cost", "mana")
        row("Pickaxe power", "pick", lambda v: f"{v}%")
        row("Axe power", "axe", lambda v: f"{v * 5}%")
        row("Hammer power", "hammer", lambda v: f"{v}%")
        row("Defense", "defense")
        row("Heals life", "healLife")
        row("Heals mana", "healMana")
        row("Shoots", "shoot", lambda v: name_of("Projectile", v))
        row("Shoot speed", "shootSpeed")
        row("Places tile", "createTile", lambda v: name_of("Tiles", v) if v >= 0 else str(v))
        row("Places wall", "createWall", lambda v: name_of("Wall", v))
        row("Buff", "buffType", lambda v: name_of("Buff", v) + (f", {d['buffTime'] // 60} s" if d.get("buffTime") else ""))
        used.add("buffTime")
        if "rare" in d:
            rows.append(("Rarity", RARITIES.get(d["rare"], str(d["rare"]))))
            used.add("rare")
        row("Value", "value", lambda v: f"buy {coins(v)}, sell {coins(v // 5)}")
        row("Max stack", "maxStack")
        for flag in ("consumable", "accessory", "vanity", "autoReuse"):
            row(flag.capitalize() if flag != "autoReuse" else "Auto reuse", flag, lambda v: "yes")
    elif data_label == "NPC":
        row("Life", "lifeMax")
        row("Damage", "damage")
        row("Defense", "defense")
        if "knockBackResist" in d:
            rows.append(("Knockback taken", f"{d['knockBackResist'] * 100:.0f}%"))
            used.add("knockBackResist")
        row("Coins dropped", "value", coins)
        for flag, title in (("boss", "Boss"), ("townNPC", "Town NPC"), ("friendly", "Friendly")):
            row(title, flag, lambda v: "yes")
        row("AI style", "aiStyle")
        row("Catch item", "catchItem", lambda v: name_of("Item", v))
    elif data_label == "Projectile":
        row("AI style", "aiStyle")
        row("Pierces", "penetrate", lambda v: "infinitely" if v < 0 else f"{v} enem" + ("y" if v == 1 else "ies"))
        row("Lifetime", "timeLeft", lambda v: f"{v / 60:.1f} s")
        side = [s for s in ("friendly", "hostile") if d.get(s)]
        kind = [c for c in DAMAGE_CLASSES if d.get(c)]
        used.update(("friendly", "hostile", *DAMAGE_CLASSES))
        if side or kind:
            rows.append(("Hits", ", ".join(side + kind)))
        row("Extra updates", "extraUpdates")
    elif data_label in ("Tile", "Wall"):
        prefix = "tile" if data_label == "Tile" else "wall"
        flags = [name[len(prefix):] for name, values in sorted(game.main_arrays.items())
                 if name.startswith(prefix) and values.get(number)]
        if flags:
            rows.append((f"{data_label} flags", ", ".join(flags)))
        color = (game.tile_colors if data_label == "Tile" else game.wall_colors).get(number)
        if color:
            rows.append(("Map color", "#{:02x}{:02x}{:02x}".format(*color)))
    if d.get("calls"):
        rows.append(("Set up with", "; ".join(d["calls"])))
    used.add("calls")
    other = {k: v for k, v in d.items() if k not in used}
    if other:
        rows.append(("Other defaults", ", ".join(f"{k}={v}" for k, v in sorted(other.items()))))
    flags = game.flags(data_label, number)
    if flags:
        rows.append(("ID sets", ", ".join(name if value is True else f"{name}={value}" for name, value in flags)))
    return rows


# --------------------------------------------------------------------------- Terraria worlds and players

class _Bin:
    """Little-endian reader for .wld / .plr (C# BinaryWriter: strings have a 7-bit-encoded byte length)."""

    def __init__(self, data, pos=0):
        self.data, self.pos = data, pos

    def take(self, fmt):
        values = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return values

    def one(self, fmt):
        return self.take(fmt)[0]

    def string(self):
        n = shift = 0
        while True:
            b = self.data[self.pos]
            self.pos += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                break
        self.pos += n
        return self.data[self.pos - n:self.pos].decode("utf-8", "replace")


GAME_MODES = {0: "Classic", 1: "Expert", 2: "Master", 3: "Journey"}
# Secret-seed flags after the game mode, with the world version that added them.
WORLD_SEEDS = ((222, "Drunk world (05162020)"), (227, "For the worthy"), (238, "Celebrationmk10"),
               (239, "The Constant"), (241, "Not the bees"), (249, "Don't dig up"), (266, "No traps"),
               (267, "Zenith (Get fixed boi)"))
WORLD_BOSSES = ("Eye of Cthulhu", "Eater of Worlds / Brain of Cthulhu", "Skeletron", "Queen Bee", "The Destroyer",
                "The Twins", "Skeletron Prime", None, "Plantera", "Golem", "King Slime", "Goblin Tinkerer saved",
                "Wizard saved", "Mechanic saved", "Goblin Army", "Clown", "Frost Legion", "Pirates")
# Map colors the game draws behind tiles: sky, underground (dirt), caverns (rock), underworld; liquids.
MAP_SKY, MAP_DIRT, MAP_ROCK, MAP_HELL = (132, 170, 248), (88, 61, 46), (74, 67, 60), (40, 20, 20)
MAP_LIQUIDS = ((0, 0, 0), (9, 61, 191), (253, 32, 3), (254, 194, 20), (190, 150, 255))  # -, water, lava, honey, shimmer
MAP_UNKNOWN_TILE = (200, 200, 200)


def world_header(data):
    """World info from the start of a .wld (version 1.3+): name, size, seed, mode, bosses... plus what reading
    the tiles needs (section pointers, which tile types store frames)."""
    r = _Bin(data)
    version = r.one("<i")
    if version < 135 or data[4:11] != b"relogic" or data[11] != 2:
        raise ValueError(f"not a Terraria 1.3+ world (version {version})")
    r.pos = 24  # past the file type, revision and flags
    pointers = r.take(f"<{r.one('<h')}i")
    n = r.one("<h")
    bits = np.frombuffer(data, np.uint8, (n + 7) // 8, r.pos)
    info = {"version": version, "pointers": pointers,
            "frame_important": np.unpackbits(bits, bitorder="little")[:n].astype(bool)}
    r.pos = pointers[0]
    info["name"] = r.string()
    if version >= 179:
        info["seed"] = str(r.one("<i")) if version == 179 else r.string()
        r.pos += 8  # world generator version
    if version >= 181:
        r.pos += 16  # GUID
    info["id"] = r.one("<i")
    r.pos += 16  # world bounds in pixels
    info["height"], info["width"] = r.take("<2i")
    mode, seeds = 0, []
    if version >= 209:
        mode = r.one("<i")
        seeds = [name for since, name in WORLD_SEEDS if version >= since and r.one("<?")]
    elif version >= 112:
        mode = int(r.one("<?"))
    info["mode"], info["seeds"] = GAME_MODES.get(mode, str(mode)), seeds
    if version >= 141:
        r.pos += 8  # creation time
    r.pos += 1 + 17 * 4  # moon type, tree / cave / ice / jungle / hell background styles
    info["spawn"] = r.take("<2i")
    info["surface"], info["rock"] = r.take("<2d")
    r.pos += 8 + 1 + 4 + 1 + 1  # time, day, moon phase, blood moon, eclipse
    info["dungeon"] = r.take("<2i")
    info["evil"] = "Crimson" if r.one("<?") else "Corruption"
    flags = r.take(f"<{len(WORLD_BOSSES) - (0 if version >= 118 else 1)}?")
    if version < 118:  # no King Slime flag yet
        flags = flags[:10] + (False,) + flags[10:]
    info["defeated"] = [name for name, done in zip(WORLD_BOSSES, flags) if name and done]
    r.pos += 2 + 1 + 4  # shadow orb smashed, meteor, orbs, altars
    info["hardmode"] = r.one("<?")
    return info


def world_layers(data, info):
    """(tiles, walls, liquids) of a world as (height, width) arrays: tile type + 1 (0 = empty), wall type,
    liquid (0 none, 1 water, 2 lava, 3 honey, 4 shimmer). Tiles are stored column by column, run-length
    encoded; every tile starts with 1-4 flag bytes saying what follows."""
    w, h, version = info["width"], info["height"], info["version"]
    important = info["frame_important"].tolist()
    n_important = len(important)
    counts, tiles, walls, liquids = [], [], [], []
    o, done, total = info["pointers"][1], 0, w * h
    u16 = struct.Struct("<H").unpack_from
    has_h4 = version >= 269
    while done < total:
        h1 = data[o]
        o += 1
        h3 = 0
        if h1 & 1:
            h2 = data[o]
            o += 1
            if h2 & 1:
                h3 = data[o]
                o += 1
                if h3 & 1 and has_h4:
                    o += 1  # invisible / full-bright flags
        tile = 0
        if h1 & 2:
            if h1 & 0x20:
                tile = u16(data, o)[0]
                o += 2
            else:
                tile = data[o]
                o += 1
            if tile < n_important and important[tile]:
                o += 4  # frame x, y
            if h3 & 8:
                o += 1  # paint
            tile += 1
        wall = 0
        if h1 & 4:
            wall = data[o]
            o += 1
            if h3 & 16:
                o += 1  # wall paint
        liquid = (h1 >> 3) & 3
        if liquid:
            o += 1  # amount
            if h3 & 0x80:
                liquid = 4
        if h3 & 0x40:
            wall |= data[o] << 8
            o += 1
        run = h1 >> 6
        count = 1
        if run == 1:
            count += data[o]
            o += 1
        elif run == 2:
            count += u16(data, o)[0]
            o += 2
        counts.append(count)
        tiles.append(tile)
        walls.append(wall)
        liquids.append(liquid)
        done += count
    n = np.array(counts)
    shape = (w, h)  # x outer, y inner -> transpose
    return tuple(np.repeat(np.array(values, dtype), n)[:total].reshape(shape).T
                 for values, dtype in ((tiles, np.uint16), (walls, np.uint16), (liquids, np.uint8)))


def world_map(info, layers, tile_colors, wall_colors):
    """The whole world as an RGB image, one pixel per tile, in map colors."""
    tiles, walls, liquids = layers
    h, w = tiles.shape
    rows = np.empty((h, 3), np.uint8)
    surface, rock = int(info["surface"]), int(info["rock"])
    rows[:surface], rows[surface:rock], rows[rock:], rows[max(0, h - 200):] = MAP_SKY, MAP_DIRT, MAP_ROCK, MAP_HELL
    rgb = np.repeat(rows[:, None, :], w, axis=1)

    def lookup(colors, size, unknown):
        table = np.empty((size, 3), np.uint8)
        table[:] = unknown
        for kind, color in colors.items():
            if 0 <= kind < size:
                table[kind] = color
        return table

    wall_table = lookup(wall_colors, int(walls.max()) + 1, MAP_DIRT)
    mask = walls > 0
    rgb[mask] = wall_table[walls[mask]]
    mask = liquids > 0
    rgb[mask] = np.array(MAP_LIQUIDS, np.uint8)[liquids[mask]]
    tile_table = lookup({k + 1: v for k, v in tile_colors.items()}, int(tiles.max()) + 1, MAP_UNKNOWN_TILE)
    mask = tiles > 0
    rgb[mask] = tile_table[tiles[mask]]
    from PIL import Image
    return Image.fromarray(rgb)


def world_rows(info):
    """Info panel rows of a world."""
    rows = [("World", info["name"]), ("Size", f"{info['width']} × {info['height']} tiles"),
            ("Mode", info["mode"] + (" (hardmode)" if info["hardmode"] else "")), ("Evil", info["evil"])]
    if info.get("seed"):
        rows.append(("Seed", info["seed"]))
    if info["seeds"]:
        rows.append(("Special seed", ", ".join(info["seeds"])))
    rows += [("Defeated", ", ".join(info["defeated"]) or "nothing yet"),
             ("Spawn", "{}, {}".format(*info["spawn"])), ("Dungeon", "{}, {}".format(*info["dungeon"])),
             ("World version", str(info["version"]))]
    return rows


PLAYER_DIFFICULTY = {0: "Classic", 1: "Mediumcore", 2: "Hardcore", 3: "Journey"}
PLAYER_COLORS = ("Hair", "Skin", "Eyes", "Shirt", "Undershirt", "Pants", "Shoes")
ARMOR_SLOTS = ["Head", "Body", "Legs"] + [f"Accessory {i}" for i in range(1, 8)] + \
    ["Vanity head", "Vanity body", "Vanity legs"] + [f"Vanity accessory {i}" for i in range(1, 8)]
BANKS = ("Piggy Bank", "Safe", "Defender's Forge", "Void Vault")


def player_data(data):
    """A .plr file decrypted (AES-128-CBC with Terraria's fixed key)."""
    return aes_cbc_decrypt(data, PLAYER_KEY, PLAYER_KEY)


def read_player(plain):
    """A decrypted player: name, difficulty, play time and - for the 1.4.4 layout (version 279+) - stats, colors,
    equipment, inventory, banks and buffs."""
    r = _Bin(plain)
    version = r.one("<i")
    if version >= 135:
        if plain[4:11] != b"relogic" or plain[11] != 3:
            raise ValueError("not a Terraria player file")
        r.pos = 24
    p = {"version": version, "name": r.string(), "difficulty": PLAYER_DIFFICULTY.get(r.one("<B"), "?")}
    if version >= 138:
        p["play_seconds"] = r.one("<q") // 10_000_000
    if version < 279:
        return p  # older layouts differ in many small places: name / difficulty only
    p["hair"] = r.one("<i")
    r.pos += 1 + 2 + 1  # hair dye, hidden accessories, hidden misc
    p["skin"] = r.one("<B")
    p["life"], p["life_max"], p["mana"], p["mana_max"] = r.take("<4i")
    r.pos += 11  # extra accessory slot, biome torches, permanent boosts, Old One's Army
    p["tax"], p["deaths"], p["pvp_deaths"] = r.take("<3i")
    p["colors"] = [r.take("<3B") for _ in PLAYER_COLORS]
    p["armor"] = [r.take("<iB") for _ in ARMOR_SLOTS]
    p["dyes"] = [r.take("<iB") for _ in range(10)]
    p["inventory"] = [r.take("<iiB?") for _ in range(58)]
    p["misc"] = [r.take("<iB") for _ in range(10)]  # pet, light pet, minecart, mount, hook + their dyes
    p["banks"] = [[r.take("<iiB") for _ in range(40)] for _ in BANKS]
    r.pos += 1  # void vault flags
    p["buffs"] = [b for b in (r.take("<ii") for _ in range(44)) if b[0] > 0]
    return p


def player_text(p, game):
    """A player as readable text (item, prefix and buff names from the game)."""
    def item(type_, prefix=0, stack=1):
        name = game.item_name(type_) if game else f"item #{type_}"
        pre = game.prefix_name(prefix) if game and prefix else ""
        return (f"{pre} {name}" if pre else name) + (f" x{stack}" if stack > 1 else "")

    hours, minutes = divmod(p.get("play_seconds", 0) // 60, 60)
    lines = [p["name"], "=" * len(p["name"]), f"Difficulty: {p['difficulty']}", f"Play time: {hours} h {minutes} min"]
    if "life" not in p:
        lines.append(f"\n(Player file version {p['version']}: only the name is read for versions before 1.4.4.)")
        return "\n".join(lines)
    lines += [f"Life: {p['life']} / {p['life_max']}", f"Mana: {p['mana']} / {p['mana_max']}",
              f"Deaths: {p['deaths']} (PvP {p['pvp_deaths']})", f"Hair style {p['hair'] + 1}, skin {p['skin']}",
              "Colors: " + ", ".join(f"{n} #{r:02x}{g:02x}{b:02x}" for n, (r, g, b) in zip(PLAYER_COLORS, p["colors"])),
              "", "Equipment:"]
    lines += [f"  {slot}: {item(t, pre)}" for slot, (t, pre) in zip(ARMOR_SLOTS, p["armor"]) if t > 0] or ["  -"]
    lines += ["", "Inventory:"]
    lines += [f"  {i + 1:2}. {item(t, pre, n)}" + (" (favorite)" if fav else "")
              for i, (t, n, pre, fav) in enumerate(p["inventory"]) if t > 0] or ["  -"]
    for bank, slots in zip(BANKS, p["banks"]):
        used = [item(t, pre, n) for t, n, pre in slots if t > 0]
        if used:
            lines += ["", f"{bank}:"] + [f"  {u}" for u in used]
    if p["buffs"]:
        def buff(t):
            found = game.lookup("Buff", t) if game else None
            return found[2] if found else f"buff #{t}"
        lines += ["", "Buffs:"] + [f"  {buff(t)} ({ticks // 60} s)" for t, ticks in p["buffs"]]
    return "\n".join(lines)


def documents_folder():
    """The user's Documents folder (moved by OneDrive or not)."""
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:  # type: ignore[attr-defined]
                return buf.value  # CSIDL_PERSONAL
        except (OSError, AttributeError):
            pass
    return os.path.join(os.path.expanduser("~"), "Documents")


def save_folders(game_path):
    """[(label, folder)] holding Worlds / Players: My Games/Terraria in Documents and Steam Cloud copies."""
    out = []
    for folder in (os.path.join(documents_folder(), "My Games", "Terraria"),
                   os.path.join(os.path.expanduser("~"), ".local", "share", "Terraria"),
                   os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Terraria")):
        if os.path.isdir(folder):
            out.append(("Documents", folder))
    parts = os.path.normpath(game_path).split(os.sep)
    low = [p.lower() for p in parts]
    if "steamapps" in low:
        userdata = os.sep.join(parts[:low.index("steamapps")] + ["userdata"])
        try:
            users = os.listdir(userdata)
        except OSError:
            users = []
        for user in users:
            remote = os.path.join(userdata, user, "105600", "remote")
            if os.path.isdir(remote):
                out.append(("Steam Cloud", remote))
    return out


def save_files(game_path):
    """[(label, "Worlds" | "Players", file path)] of the saved worlds and players (backups left out)."""
    found = []
    for label, folder in save_folders(game_path):
        names = listdir_lower(folder)
        for sub, ext in (("worlds", ".wld"), ("players", ".plr")):
            if sub in names:
                path = os.path.join(folder, names[sub])
                for n in sorted(os.listdir(path)):
                    if n.lower().endswith(ext):
                        found.append((label, sub.capitalize(), os.path.join(path, n)))
    return found


def find_exe(path, content):
    """Terraria.exe next to the Content folder, or None."""
    for folder in (path, os.path.dirname(content)):
        exe = listdir_lower(folder).get("terraria.exe")
        if exe:
            return os.path.join(folder, exe)
    return None


def terraria_assets(session, exe):
    """Names and animations from Terraria.exe. Item_1 -> "Item: Iron Pickaxe (1)", Music_1 -> "Music: Overworld
    Day (1)": the path changes too, so exported files get the name; the uid stays the file's (favorites keep
    working). Animated NPC / projectile / item sheets also get an "Animations/..." entry (plays in the viewer,
    exports as a GIF)."""
    try:
        with open(exe, "rb") as f:
            game = TerrariaData(f.read())
    except (OSError, DotNetError) as e:
        log.warning("Terraria names from %s: %s", exe, e)
        return
    session.terraria = game
    animations = []
    for asset in session.assets:
        if not (asset.uid.startswith("images/") or asset.ref[0] == "xwb"):
            continue  # Sounds/Item_1 is a sound ID, not an item
        folder, _sep, base = asset.name.rpartition("/")
        m = _NUMBERED.match(base)
        number = int(m.group(2)) if m else -1
        found = game.lookup(m.group(1), number) if m else None
        if found:
            label, internal, shown = found
            session.names[asset.uid] = (label, number, internal, shown)
            base = f"{label}: {shown} ({number})"
            asset.name = asset.path = f"{folder}/{base}" if folder else base
        anim = animation_of(game, m.group(0)) if m and asset.kind == "texture" else None
        if anim:
            name = f"Animations/{base}"
            animations.append(Asset("animation", name, ("anim", asset.uid), uid="anim:" + asset.uid, path=name,
                                    source=asset.source, ref=("anim", asset, *anim), ext="gif"))
    session.assets += animations
    add_saves(session)


def add_saves(session):
    """Saved worlds (shown as their map) and players (shown as text) from Documents/My Games/Terraria and Steam
    Cloud."""
    for label, kind, path in save_files(session.path):
        cloud = " (Steam Cloud)" if label == "Steam Cloud" else ""
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                head = f.read(1 << 16)
            if kind == "Worlds":
                name = world_header(head)["name"]
                asset = Asset("texture", f"Worlds/{name}{cloud}", ("wld", path.lower()), uid="wld:" + path.lower(),
                              size=size, source=os.path.basename(path), ref=("wld", path), ext="wld")
            else:
                name = read_player(player_data(head))["name"]
                asset = Asset("data", f"Players/{name}{cloud}", ("plr", path.lower()), uid="plr:" + path.lower(),
                              size=size, source=os.path.basename(path), ref=("plr", path), ext="txt")
        except (OSError, ValueError, IndexError, struct.error) as e:
            log.warning("Terraria save %s: %s", path, e)
            continue
        asset.path = asset.name
        session.assets.append(asset)
        session.file_count += 1


# --------------------------------------------------------------------------- the plugin

# Folder words that say what a compressed .xnb holds (the type is only known after decompressing it).
SOUND_DIRS = re.compile(r"(^|/)(sounds?|sfx|audio|music)(/|$)", re.I)
FILE_DIRS = re.compile(r"(^|/)(effects?|shaders?|maps|data|strings|localization|dialogue)(/|$)|shader", re.I)
READER_KINDS = {"Texture2DReader": "texture", "SpriteFontReader": "texture", "DynamicSpriteFontReader": "texture",
                "SoundEffectReader": "audio"}


def content_dir(path):
    """The Content folder of an XNA / FNA game folder (or the folder itself), or None."""
    if os.path.isdir(path):
        names = listdir_lower(path)
        if "content" in names and os.path.isdir(os.path.join(path, names["content"])):
            return os.path.join(path, names["content"])
        if any(n.endswith(".xnb") for n in names):
            return path
    return None


def _has_xnb(folder, depth=2):
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return False
    if any(e.name.lower().endswith(".xnb") for e in entries if e.is_file()):
        return True
    return depth > 0 and any(_has_xnb(e.path, depth - 1) for e in entries if e.is_dir())


def guess_kind(rel, head):
    """Asset kind of an .xnb from its first bytes: uncompressed files name their reader right there; for
    compressed ones the folder decides (Images -> texture, Sounds -> audio, Effects -> file)."""
    try:
        _platform, _version, compressed, _hidef = xnb_header(head)
    except XnbError:
        return "file"
    if not compressed:
        try:
            _readers, main, _r = xnb_object(head[10:])
            return READER_KINDS.get(main, "file")
        except (IndexError, ValueError, UnicodeDecodeError):
            pass
    if SOUND_DIRS.search(rel):
        return "audio"
    if FILE_DIRS.search(rel):
        return "file"
    return "texture"


class XnaSession(GameSession):
    def __init__(self, plugin, path):
        super().__init__(plugin, path)
        self.content = ""
        self._banks = {}  # .xwb path -> open file
        self.names = {}  # asset uid -> (label, ID, internal name, display name) from Terraria.exe
        self.terraria = None  # TerrariaData when Terraria.exe was read
        self._sheet: tuple = (None, None)  # (uid, PIL image) of the last sheet frames were cut from
        self._world: tuple = (None, None)  # (path, (header, map image)) of the last world drawn

    def close(self):
        for f in self._banks.values():
            f.close()
        self._banks.clear()

    def _world_map(self, path):
        if self._world[0] != path:
            with open(path, "rb") as f:
                data = f.read()
            info = world_header(data)
            game = self.terraria
            image = world_map(info, world_layers(data, info), game.tile_colors if game else {},
                              game.wall_colors if game else {})
            self._world = (path, (info, image))
        return self._world[1]

    def _player_text(self, path):
        with open(path, "rb") as f:
            return player_text(read_player(player_data(f.read())), self.terraria)

    def raw(self, asset):
        if asset.ref[0] == "plr":
            return self._player_text(asset.ref[1]).encode("utf-8")
        if asset.ref[0] == "xwb":
            return self.audio(asset)[0]
        if asset.ref[0] == "anim":
            frames, length = self.sprite_frames(asset)
            images = [self.image(frame).convert("RGBA") for _t, frame in frames]
            return gif_bytes(images, length / len(images))
        with open(asset.ref[1], "rb") as f:
            return f.read()

    def image(self, asset):
        if asset.ref[0] == "wld":
            return self._world_map(asset.ref[1])[1]
        if asset.ref[0] == "frame":
            _tag, sheet, (x, y, w, h) = asset.ref
            return self._sheet_image(sheet).crop((x, y, x + w, y + h))
        return xnb_image(self.raw(asset))

    def _sheet_image(self, sheet):
        if self._sheet[0] != sheet.uid:
            self._sheet = (sheet.uid, xnb_image(self.raw(sheet)))
        return self._sheet[1]

    def _frames(self, sheet):
        """Terraria sheet frames [(x, y, w, h)], y from the top."""
        if self.terraria is None or sheet.kind != "texture" or sheet.ref[0] != "xnb":
            return []
        w, h, _fmt = texture_info(self.raw(sheet))
        stem = os.path.splitext(os.path.basename(sheet.ref[1]))[0]
        return sheet_frames(self.terraria, stem, w, h)

    def sprite_rects(self, asset):
        frames = self._frames(asset)
        if not frames:
            return []
        h = texture_info(self.raw(asset))[1]
        return [(x, h - y - fh, fw, fh) for x, y, fw, fh in frames]

    def animation_dim(self, asset):
        return "2d"  # every animation here is a flipbook

    def sprite_regions(self, asset):
        if asset.ref[0] != "frame":
            return []
        _tag, sheet, (x, y, w, h) = asset.ref
        sheet_h = texture_info(self.raw(sheet))[1]
        return [(sheet, (x, sheet_h - y - h, w, h), None)]

    def sprite_frames(self, asset):
        if asset.ref[0] != "anim":
            return [], 0.0
        _tag, sheet, count, seconds, ping_pong = asset.ref
        rects = self._frames(sheet)
        if len(rects) != count:
            return [], 0.0
        frames = []
        for i, index in enumerate(frame_order(count, ping_pong)):
            frame = Asset("sprite", f"{asset.name} #{index}", ("frame", sheet.uid, index), uid=f"{sheet.uid}#{index}",
                          source=sheet.source, ref=("frame", sheet, rects[index]))
            frames.append((i * seconds, frame))
        return frames, len(frames) * seconds

    def text(self, asset):
        if asset.ref[0] == "anim":
            _tag, sheet, count, seconds, ping_pong = asset.ref
            return (f"{asset.name}\n\n{count} frames from {sheet.name}, {seconds * 1000:.0f} ms each"
                    + (", ping-pong" if ping_pong else "") + ".\nExport saves it as an animated GIF.")
        if asset.ref[0] == "plr":
            return self._player_text(asset.ref[1])
        return super().text(asset)

    def audio(self, asset):
        if asset.ref[0] == "xwb":
            _tag, bank, entry = asset.ref
            if bank not in self._banks:
                self._banks[bank] = open(bank, "rb")
            return xwb_wav(self._banks[bank], entry), "wav"
        return xnb_sound(self.raw(asset)), "wav"

    def stats(self, asset):
        stats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        if asset.kind == "texture" and asset.ref[0] == "xnb":
            try:
                w, h, _fmt = texture_info(self.raw(asset))
                stats.update(w=w, h=h, info=f"{w}×{h}", sort=w * h)
            except Exception:
                pass
        elif asset.ref[0] == "xwb":
            _i, codec, channels, rate, _align, _off, length = asset.ref[2]
            stats["info"] = f"{CODECS.get(codec, codec)}, {channels} ch, {rate} Hz"
        elif asset.ref[0] == "wld":
            try:
                with open(asset.ref[1], "rb") as f:
                    info = world_header(f.read(1 << 16))
                w, h = info["width"], info["height"]
                stats.update(w=w, h=h, info=f"{w}×{h} tiles", sort=w * h)
            except (OSError, ValueError, IndexError, struct.error):
                pass
        elif asset.ref[0] == "anim":
            stats.update(info=f"{asset.ref[2]} frames", sort=asset.ref[2])
        return stats

    def describe(self, asset):
        rows = []
        if asset.uid in self.names:
            label, number, internal, shown = self.names[asset.uid]
            rows = [("Name", shown), (f"{label} ID", f"{number} ({internal})")] + game_rows(self.terraria, label, number)
        return rows + self._describe_file(asset)

    def _describe_file(self, asset):
        if asset.ref[0] == "wld":
            with open(asset.ref[1], "rb") as f:
                return world_rows(world_header(f.read(1 << 16))) + [("File", asset.ref[1])]
        if asset.ref[0] == "plr":
            with open(asset.ref[1], "rb") as f:
                p = read_player(player_data(f.read()))
            return [("Player", p["name"]), ("Difficulty", p["difficulty"]), ("Player version", str(p["version"])),
                    ("File", asset.ref[1])]
        if asset.ref[0] == "anim":
            _tag, sheet, count, seconds, ping_pong = asset.ref
            return [("Sheet", sheet.name), ("Frames", f"{count}" + (" (ping-pong)" if ping_pong else "")),
                    ("Frame time", f"{seconds * 1000:.0f} ms")]
        if asset.ref[0] == "xwb":
            _i, codec, channels, rate, _align, offset, length = asset.ref[2]
            return [("File", os.path.basename(asset.ref[1])), ("Entry", str(asset.ref[2][0])),
                    ("Format", f"{CODECS.get(codec, codec)}, {channels} channel(s), {rate} Hz"),
                    ("Size", f"{length:,} bytes")]
        rows = [("File", os.path.relpath(asset.ref[1], self.content).replace("\\", "/"))]
        try:
            data = self.raw(asset)
            platform, version, compressed, hidef = xnb_header(data)
            readers, main, _r = xnb_object(xnb_content(data))
            rows += [("Content type", main.replace("Reader", "")),
                     ("XNB", f"version {version}, platform '{platform}'" + (", LZX compressed" if compressed else "")
                      + (", HiDef" if hidef else ""))]
            if len(readers) > 1:
                rows.append(("Readers", ", ".join(_short(r) for r in readers)))
        except Exception as e:
            rows.append(("Error", str(e)))
        return rows


class XnaPlugin(EnginePlugin):
    id = "xna"
    name = "Terraria / XNA / FNA"
    version = "1.0"
    author = "UniView"
    description = ("XNA and FNA games (Terraria, Stardew Valley, Celeste...): .xnb textures, sprite-font sheets "
                   "and sounds, XACT music banks (.xwb).")

    def detect(self, path):
        folder = content_dir(path)
        if folder is None:
            return 0
        names = listdir_lower(path) if os.path.isdir(path) else {}
        if "terraria.exe" in names or "terraria" in names:
            return 95
        return 85 if _has_xnb(folder) else 0

    def game_info(self, path):
        names = listdir_lower(path) if os.path.isdir(path) else {}
        version = ""
        if "changelog.txt" in names:  # Terraria: "Version 1.4.5.8 Changes ---"
            try:
                with open(os.path.join(path, names["changelog.txt"]), encoding="utf-8", errors="replace") as f:
                    m = re.match(r"\s*Version\s+([\d.]+)", f.readline())
                    version = m.group(1) if m else ""
            except OSError:
                pass
        framework = "FNA" if "fna.dll" in names else "XNA 4"
        return {"engine_version": "", "detail": framework + (f", game version {version}" if version else "")}

    def count_files(self, path):
        folder = content_dir(path)
        return sum(len(files) for _r, _d, files in os.walk(folder)) if folder else 0

    def open(self, path, progress):
        session = XnaSession(self, path)
        folder = content_dir(path)
        if folder is None:
            raise FileNotFoundError(f"No XNA Content folder found in:\n{path}")
        session.content = folder
        progress("Listing content ...")
        banks, cues = [], {}
        for dirpath, _dirs, names in os.walk(folder):
            for n in sorted(names):
                full = os.path.join(dirpath, n)
                rel = os.path.relpath(full, folder).replace("\\", "/")
                low = n.lower()
                if low.endswith(".xwb"):
                    banks.append((rel, full))
                    continue
                if low.endswith(".xsb"):
                    try:
                        with open(full, "rb") as f:
                            cues[rel] = xsb_cues(f.read())
                    except (OSError, ValueError) as e:
                        log.debug("Sound bank %s: %s", rel, e)
                    continue
                if not low.endswith(".xnb"):
                    continue
                try:
                    with open(full, "rb") as f:
                        head = f.read(1024)
                    size = os.path.getsize(full)
                except OSError:
                    continue
                kind = guess_kind(rel, head)
                stem = rel[:-4]
                session.assets.append(Asset(kind, stem, ("xnb", rel.lower()), uid=rel.lower(), size=size, path=stem,
                                            source=os.path.dirname(rel) or "Content", ref=("xnb", full),
                                            ext="xnb"))
        cue_names = next(iter(cues.values())) if len(cues) == 1 and len(banks) == 1 else {}
        for rel, full in banks:
            try:
                with open(full, "rb") as f:
                    entries = xwb_entries(f.read(1 << 20))
            except (OSError, ValueError) as e:
                log.warning("Wave bank %s: %s", rel, e)
                continue
            stem = os.path.splitext(rel)[0]
            for entry in entries:
                cue = cue_names.get(entry[0])  # Terraria: "Music_1" is what the game's code plays
                name = f"Music/{cue}" if cue else f"{stem}/{entry[0] + 1:03d}"
                session.assets.append(Asset("audio", name, ("xwb", rel.lower(), entry[0]),
                                            uid=f"{rel.lower()}#{entry[0]}", size=entry[6], path=name,
                                            source=os.path.basename(rel), ref=("xwb", full, entry), ext="wav"))
        session.file_count = len(session.assets)
        if not session.assets:
            raise FileNotFoundError(f"No .xnb or .xwb files found in:\n{folder}")
        exe = find_exe(path, folder)
        if exe:
            progress("Reading names and animations from Terraria.exe ...")
            terraria_assets(session, exe)
        return session



PLUGIN = XnaPlugin()
