"""Reading Unreal packages (.uasset/.umap): class of each export, texture and mesh data.

Two package formats:
* Zen packages (IoStore .utoc/.ucas, UE5 and late UE4): header with a name map and an export map;
  classes are "script imports" resolved through the global ScriptObjects table (global.utoc).
* Legacy packages (.pak files): .uasset header (FPackageFileSummary) + .uexp export data + .ubulk.

Cooked games store object properties "unversioned" (no names or sizes without the game's
mappings file), so textures and meshes are found by the structures that follow the properties:
* Texture2D: the platform data starts with SizeX, SizeY, PackedData and the pixel format name as a
  string ("PF_DXT5" ...), followed by the mips and their bulk data.
* StaticMesh / SkeletalMesh: the position buffer is "stride 12, N vertices, element size 12, N"
  followed by the vertex positions, then the tangent and UV buffers; the index buffer is found by
  checking that every index is < N.
"""

import re
import struct

import numpy as np

from engines.sdk import MeshData, orient_to_normals

LEGACY_TAG = 0x9E2A83C1
NAME_HASH_VERSION = 0xC1640000  # marks a serialized name batch

# FPackageObjectIndex type (top 2 bits)
INDEX_EXPORT, INDEX_SCRIPT_IMPORT, INDEX_PACKAGE_IMPORT, INDEX_NULL = 0, 1, 2, 3

TEXTURE_CLASSES = {"Texture2D", "TextureCube", "Texture2DArray", "LightMapTexture2D", "ShadowMapTexture2D",
                   "VirtualTexture2D", "TextureRenderTarget2D", "VolumeTexture", "TextureCubeArray"}
MESH_CLASSES = {"StaticMesh", "SkeletalMesh"}
# Classes shown as "the" class of a package, most interesting first.
CLASS_PRIORITY = ["Texture2D", "TextureCube", "VolumeTexture", "Texture2DArray", "StaticMesh", "SkeletalMesh",
                  "SoundWave", "SoundCue", "Material", "MaterialInstanceConstant", "AnimSequence", "Skeleton",
                  "PhysicsAsset", "DataTable", "Blueprint", "BlueprintGeneratedClass", "WidgetBlueprintGeneratedClass",
                  "World", "Font", "FontFace", "NiagaraSystem", "ParticleSystem", "CurveFloat", "StringTable"]


# --------------------------------------------------------------------------- name maps

def read_name_batch(data, pos):
    """UE5 serialized name batch -> ([names], position after)."""
    count, = struct.unpack_from("<I", data, pos)
    pos += 4
    if count == 0:
        return [], pos
    _string_bytes, = struct.unpack_from("<I", data, pos)
    pos += 4 + 8 + count * 8  # hash version + hashes
    headers = data[pos:pos + count * 2]
    pos += count * 2
    names = []
    for i in range(count):
        b0, b1 = headers[i * 2], headers[i * 2 + 1]
        utf16 = b0 & 0x80
        length = ((b0 & 0x7F) << 8) | b1
        if utf16:
            pos += pos & 1
            names.append(data[pos:pos + length * 2].decode("utf-16-le", "replace"))
            pos += length * 2
        else:
            names.append(data[pos:pos + length].decode("latin-1"))
            pos += length
    return names, pos


def mapped_name(names, index, number=0):
    index &= 0x3FFFFFFF
    name = names[index] if index < len(names) else f"name{index}"
    return f"{name}_{number - 1}" if number else name


def read_script_objects(data):
    """global.utoc ScriptObjects chunk -> {FPackageObjectIndex value: object name}."""
    names, pos = read_name_batch(data, 0)
    count, = struct.unpack_from("<i", data, pos)
    pos += 4
    out = {}
    for i in range(count):
        name_index, number, global_index = struct.unpack_from("<IIQ", data, pos + i * 32)
        out[global_index] = mapped_name(names, name_index, number)
    return out


# --------------------------------------------------------------------------- package headers

class PackageInfo:
    """What we know about a package: its exports' (name, class) and where the export data starts."""

    def __init__(self, zen, names, exports, data_offset, header_size=0, bulk_map=None, imported_packages=None):
        self.imported_packages = imported_packages or []  # "/Game/Path/Name" of packages it uses
        self.zen = zen
        self.names = names
        self.exports = exports        # [(object name, class name)]
        self.data_offset = data_offset
        self.header_size = header_size
        self.bulk_map = bulk_map or []  # UE5.2+: [(serial offset, serial size, flags)]

    @property
    def main_class(self):
        classes = [c for _n, c in self.exports if c]
        for cls in CLASS_PRIORITY:
            if cls in classes:
                return cls
        return classes[0] if classes else ""


def zen_header_size(data):
    return struct.unpack_from("<I", data, 4)[0]


def parse_zen(data, script_objects):
    """Zen package header (UE5). data must hold at least the header."""
    has_versioning, header_size, _n_index, _n_number, _flags, cooked_header_size = struct.unpack_from("<6I", data, 0)
    imported_hashes, import_map, export_map, bundle_entries = struct.unpack_from("<4i", data, 24)
    best = None
    for summary_size in (44, 52, 60):  # UE5.0-5.2, UE5.3-5.4, UE5.5+
        pos = summary_size
        try:
            if has_versioning:
                pos += 4 + 8 + 4  # zen version, file version UE4/UE5, licensee version
                n, = struct.unpack_from("<i", data, pos)
                pos += 4 + n * 20
            count, = struct.unpack_from("<I", data, pos)
            if count and struct.unpack_from("<Q", data, pos + 8)[0] != NAME_HASH_VERSION:
                continue
            names, after = read_name_batch(data, pos)
        except (struct.error, UnicodeDecodeError, IndexError):
            continue
        # UE5.2+: a bulk data map sits between the names and the imported hashes, as
        # [u64 size][entries] (5.2) or [u64 pad size][pad][u64 size][entries] (5.3+).
        bulk_map = None
        rest = imported_hashes - after
        starts = [after]
        if rest >= 16:
            pad, = struct.unpack_from("<Q", data, after)
            if pad < 8:
                starts.insert(0, after + 8 + pad)
        for start in starts:
            if imported_hashes - start < 8:
                continue
            size, = struct.unpack_from("<Q", data, start)
            if size % 32 == 0 and start + 8 + size <= imported_hashes and imported_hashes - (start + 8 + size) < 16:
                bulk_map = []
                for i in range(size // 32):
                    offset, _dup, serial_size, flags = struct.unpack_from("<qqqI", data, start + 8 + i * 32)
                    bulk_map.append((offset, serial_size, flags))
                break
        if bulk_map is not None or 0 <= rest < 16:
            best = (names, bulk_map or [], summary_size)
            break
    if best is None:
        raise ValueError("Unrecognized package header")
    names, bulk_map, summary_size = best
    imported_packages = []
    if summary_size >= 52:  # UE5.3+: names of the packages this one imports from
        names_offset, = struct.unpack_from("<i", data, 48)
        if 0 < names_offset < min(len(data), header_size):
            try:
                imported_packages, _end = read_name_batch(data, names_offset)
            except (struct.error, IndexError, UnicodeDecodeError):
                imported_packages = []
    exports = []
    count = max(0, (bundle_entries - export_map) // 72)
    for i in range(count):
        e = export_map + i * 72
        _off, _size, name_i, name_n, _outer, class_index = struct.unpack_from("<QQIIQQ", data, e)
        cls = ""
        kind = class_index >> 62
        if kind == INDEX_SCRIPT_IMPORT:
            cls = script_objects.get(class_index, "")
        exports.append((mapped_name(names, name_i, name_n), cls))
    return PackageInfo(True, names, exports, header_size, header_size, bulk_map, imported_packages)


def _fstring(data, pos):
    n, = struct.unpack_from("<i", data, pos)
    pos += 4
    if n < 0:
        return data[pos:pos - n * 2].decode("utf-16-le", "replace").rstrip("\0"), pos - n * 2
    return data[pos:pos + n].decode("latin-1").rstrip("\0"), pos + n


def parse_legacy(data):
    """.uasset header (cooked UE4/UE5). Only what's needed: names, exports' classes."""
    tag, legacy = struct.unpack_from("<Ii", data, 0)
    if tag != LEGACY_TAG:
        raise ValueError("Not an Unreal package")
    pos = 8
    if legacy != -4:
        pos += 4  # UE3 version
    ue4, = struct.unpack_from("<i", data, pos)
    pos += 4
    ue5 = 0
    if legacy <= -8:
        ue5, = struct.unpack_from("<i", data, pos)
        pos += 4
    pos += 4  # licensee
    n_custom, = struct.unpack_from("<i", data, pos)
    pos += 4 + n_custom * 20
    total_header, = struct.unpack_from("<i", data, pos)
    pos += 4
    _folder, pos = _fstring(data, pos)
    flags, name_count, name_offset = struct.unpack_from("<Iii", data, pos)
    pos += 12
    if ue5 >= 1008:
        pos += 8  # soft object paths
    if not flags & 0x80000000 and ue4 >= 516:  # PKG_FilterEditorOnly unset: localization id
        _loc, pos = _fstring(data, pos)
    if ue4 >= 459:
        pos += 8  # gatherable text
    export_count, export_offset, import_count, import_offset = struct.unpack_from("<4i", data, pos)
    names = []
    p = name_offset
    for _ in range(name_count):
        name, p = _fstring(data, p)
        if ue4 >= 504:
            p += 4
        names.append(name)

    def fname(offset):
        index, number = struct.unpack_from("<ii", data, offset)
        return mapped_name(names, index, number) if 0 <= index < len(names) else ""

    imports = []
    if import_count:
        stride = (export_offset - import_offset) // import_count if export_offset > import_offset else 28
        stride = stride if 28 <= stride <= 48 else 28
        for i in range(import_count):
            base = import_offset + i * stride
            imports.append(fname(base + 20))  # ObjectName after ClassPackage, ClassName, OuterIndex
    exports = []
    # Export entries have a version-dependent size; the class index is their first field, and the
    # object name follows ClassIndex, SuperIndex, [TemplateIndex], OuterIndex.
    stride = _legacy_export_stride(ue4, ue5)
    for i in range(export_count):
        base = export_offset + i * stride
        if base + 4 > len(data):
            break
        class_index, = struct.unpack_from("<i", data, base)
        cls = imports[-class_index - 1] if class_index < 0 and -class_index - 1 < len(imports) else ""
        name_at = base + (16 if ue4 >= 508 else 12)
        exports.append((fname(name_at) if name_at + 8 <= len(data) else "", cls))
    packages = [name for name in imports if name.startswith("/")]
    return PackageInfo(False, names, exports, total_header, total_header, imported_packages=packages)


def _legacy_export_stride(ue4, ue5):
    size = 4 + 4 + 4 + 4 + 8 + 4  # class, super, template, outer, name, flags
    size += 8 + 8  # serial size, serial offset (64-bit since 4.?)
    size += 4 * 3  # forced export, not for client, not for server
    if ue5 < 1010:
        size += 16  # package guid (removed in UE5)
    if ue5 >= 1006:
        size += 4  # is inherited instance
    size += 4  # package flags
    size += 4  # not always loaded for editor game
    size += 4  # is asset
    if ue5 >= 1003:
        size += 4  # generate public hash
    size += 4 * 5  # first export dependency + 4 counts
    if ue5 >= 1011:
        size += 8 * 2  # script serialization start/end
    return size


def package_info(data, script_objects=None):
    if len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == LEGACY_TAG:
        return parse_legacy(data)
    return parse_zen(data, script_objects or {})


# --------------------------------------------------------------------------- textures

PIXEL_FORMATS = {
    "PF_DXT1": ("bcn", 1, 8), "PF_DXT3": ("bcn", 2, 16), "PF_DXT5": ("bcn", 3, 16), "PF_BC4": ("bcn", 4, 8),
    "PF_BC5": ("bcn", 5, 16), "PF_BC6H": ("bcn", 6, 16), "PF_BC7": ("bcn", 7, 16),
    "PF_B8G8R8A8": ("raw", "BGRA", 4), "PF_R8G8B8A8": ("raw", "RGBA", 4), "PF_A8R8G8B8": ("raw", "ARGB", 4),
    "PF_G8": ("raw", "L", 1), "PF_L8": ("raw", "L", 1), "PF_A8": ("raw", "L", 1), "PF_R8": ("raw", "L", 1),
    "PF_R8G8": ("raw2", "", 2), "PF_V8U8": ("raw2", "", 2),
    "PF_G16": ("num", "<u2", 2), "PF_R16F": ("num", "<f2", 2), "PF_FloatRGBA": ("num4", "<f2", 8),
    "PF_R16G16B16A16_UNORM": ("num4", "<u2", 8), "PF_A16B16G16R16": ("num4", "<u2", 8),
    "PF_R32_FLOAT": ("num", "<f4", 4), "PF_A32B32G32R32F": ("num4", "<f4", 16),
    "PF_FloatRGB": ("r11g11b10", "", 4), "PF_FloatR11G11B10": ("r11g11b10", "", 4),
    "PF_B5G5R5A1_UNORM": ("x", "", 2),
}
ASTC_RE = re.compile(r"PF_ASTC_(\d+)x(\d+)")


def mip_bytes(fmt, w, h):
    if fmt in PIXEL_FORMATS:
        kind, _a, size = PIXEL_FORMATS[fmt]
        if kind == "bcn":
            return ((w + 3) // 4) * ((h + 3) // 4) * size
        return w * h * size
    m = ASTC_RE.match(fmt)
    if m:
        bw, bh = int(m.group(1)), int(m.group(2))
        return ((w + bw - 1) // bw) * ((h + bh - 1) // bh) * 16
    if fmt.startswith("PF_ETC2_RGBA"):
        return ((w + 3) // 4) * ((h + 3) // 4) * 16
    if fmt.startswith("PF_ETC"):
        return ((w + 3) // 4) * ((h + 3) // 4) * 8
    raise ValueError(f"Pixel format {fmt} isn't supported yet")


def decode_pixels(fmt, data, w, h):
    from PIL import Image
    if fmt in PIXEL_FORMATS:
        kind, arg, _size = PIXEL_FORMATS[fmt]
        if kind == "bcn":
            pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
            mode = {4: "L", 5: "RGB", 6: "RGB"}.get(arg, "RGBA")
            args = (arg, "BC6H") if arg == 6 else (arg,)
            img = Image.frombytes(mode, (pw, ph), data, "bcn", *args)
            if arg == 5:  # normal map: rebuild blue
                a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                a[..., 2] = np.sqrt(np.clip(1 - a[..., 0] ** 2 - a[..., 1] ** 2, 0, 1))
                img = Image.fromarray(((a + 1) * 127.5).clip(0, 255).astype(np.uint8), "RGB")
            return img.crop((0, 0, w, h)) if (pw, ph) != (w, h) else img
        if kind == "raw":
            mode = "L" if arg == "L" else "RGBA"
            return Image.frombytes(mode, (w, h), data, "raw", arg)
        if kind == "raw2":
            a = np.frombuffer(data, np.uint8, w * h * 2).reshape(h, w, 2)
            return Image.fromarray(np.dstack([a, np.zeros((h, w), np.uint8)]), "RGB")
        if kind in ("num", "num4"):
            ch = 4 if kind == "num4" else 1
            a = np.frombuffer(data, arg, w * h * ch).reshape(h, w, ch).astype(np.float32)
            a = a / 65535.0 if arg == "<u2" else a / (1.0 + np.abs(a))
            rgb = a[..., :3] if ch == 4 else np.repeat(a, 3, axis=2)
            return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
        if kind == "r11g11b10":
            v = np.frombuffer(data, "<u4", w * h).reshape(h, w)

            def f(bits, mant):
                e = (bits >> mant).astype(np.int32)
                m = (bits & ((1 << mant) - 1)).astype(np.float32)
                return np.where(e == 0, m / (1 << mant) * 2.0 ** -14, (1 + m / (1 << mant)) * 2.0 ** (e - 15))
            rgb = np.dstack([f(v & 0x7FF, 6), f((v >> 11) & 0x7FF, 6), f((v >> 22) & 0x3FF, 5)])
            rgb = rgb / (1 + rgb)
            return Image.fromarray((np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), "RGB")
    m = ASTC_RE.match(fmt)
    if m or fmt.startswith("PF_ETC"):
        import texture2ddecoder as t2d
        if m:
            raw = t2d.decode_astc(data, w, h, int(m.group(1)), int(m.group(2)))
        elif fmt.startswith("PF_ETC2_RGBA"):
            raw = t2d.decode_etc2a8(data, w, h)
        elif fmt.startswith("PF_ETC2"):
            raw = t2d.decode_etc2(data, w, h)
        else:
            raw = t2d.decode_etc1(data, w, h)
        return Image.frombytes("RGBA", (w, h), raw, "raw", "BGRA")
    raise ValueError(f"Pixel format {fmt} isn't supported yet")


BULK_INLINE = 0x40
BULK_SEPARATE_FILE = 0x100
BULK_OPTIONAL = 0x800
BULK_MEMORY_MAPPED = 0x1000
BULK_SIZE64 = 0x2000
BULK_BAD_DATA_VERSION = 0x8000
BULK_UNUSED = 0x20


class Mip:
    __slots__ = ("w", "h", "flags", "size", "offset", "inline")

    def __init__(self, w, h, flags, size, offset, inline=None):
        self.w, self.h, self.flags, self.size, self.offset, self.inline = w, h, flags, size, offset, inline


def _read_bulk(data, pos, info):
    """FByteBulkData header at pos -> (flags, size, offset, inline data or None, position after)."""
    if info is not None and info.bulk_map:
        index, = struct.unpack_from("<i", data, pos)
        if not 0 <= index < len(info.bulk_map):
            raise ValueError("bad bulk index")
        offset, size, flags = info.bulk_map[index]
        pos += 4
        inline = None
        if flags & BULK_INLINE:
            inline = data[pos:pos + size]
            pos += size
        return flags, size, offset, inline, pos
    flags, = struct.unpack_from("<I", data, pos)
    pos += 4
    if flags & ~0x3FFFFFF and not flags & 0x80000000:
        raise ValueError("bad bulk flags")
    if flags & BULK_SIZE64:
        count, size = struct.unpack_from("<qq", data, pos)
        pos += 16
    else:
        count, size = struct.unpack_from("<iI", data, pos)
        pos += 8
    offset, = struct.unpack_from("<q", data, pos)
    pos += 8
    if flags & BULK_BAD_DATA_VERSION:
        pos += 2
    if size < 0 or count < 0 or size > 1 << 31:
        raise ValueError("bad bulk size")
    inline = None
    if flags & BULK_INLINE:
        inline = data[pos:pos + size]
        pos += size
    return flags, size, offset, inline, pos


PF_RE = re.compile(rb"(?s)(.)\x00\x00\x00(PF_[A-Za-z0-9_]{2,40})\x00")


def find_texture(data, info=None):
    """(pixel format, width, height, [Mip], is cube) from a Texture2D's export data, or None."""
    for m in PF_RE.finditer(data):
        fmt = m.group(2).decode("ascii")
        if m.group(1)[0] != len(fmt) + 1 or m.start() < 12:
            continue
        w, h, packed = struct.unpack_from("<iiI", data, m.start() - 12)
        if not (0 < w <= 16384 and 0 < h <= 16384):
            continue
        pos = m.end()
        if packed & (1 << 30):  # optional data
            pos += 8
        if packed & (1 << 29):  # CPU copy - not supported
            continue
        for mip_variant in (0, 1, 2):
            try:
                mips = _read_mips(data, pos, w, h, info, mip_variant)
            except (ValueError, struct.error):
                continue
            if mips:
                return fmt, w, h, mips, bool(packed & (1 << 31))
    return None


def _read_mips(data, pos, w, h, info, variant):
    """variant 0: UE5 (no bCooked), 1: UE4 (bCooked bool before each mip), 2: skip a UE5.? placeholder bool."""
    if variant == 2:
        pos += 4
    first, count = struct.unpack_from("<ii", data, pos)
    pos += 8
    if not (0 <= first < 16 and 0 < count <= 16):
        raise ValueError("bad mip count")
    mips = []
    for _ in range(count):
        if variant == 1:
            pos += 4
        flags, size, offset, inline, pos = _read_bulk(data, pos, info)
        mw, mh, _md = struct.unpack_from("<iii", data, pos)
        pos += 12
        if not (0 < mw <= w and 0 < mh <= h):
            raise ValueError("bad mip size")
        mips.append(Mip(mw, mh, flags, size, offset, inline))
    if mips[0].w != w and mips[0].w != max(1, w >> first):
        raise ValueError("mip sizes don't match")
    return mips


def _morton(v):
    v &= 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    return (v | (v << 1)) & 0x55555555


VT_RAW_GPU, VT_ZIPPED_GPU = 4, 5
MAX_VT_PIXELS = 8192 * 8192


def find_virtual_texture(data, info):
    """UE5 virtual texture (FVirtualTextureBuiltData) after the pixel format, or None."""
    for m in PF_RE.finditer(data):
        fmt = m.group(2).decode("ascii")
        if m.group(1)[0] != len(fmt) + 1:
            continue
        start = m.end()
        for p in range(start, start + 64, 4):
            try:
                layers, wb, hb, tile, border, n_off = struct.unpack_from("<6I", data, p)
            except struct.error:
                break
            if not (1 <= layers <= 8 and 1 <= wb <= 256 and 1 <= hb <= 256 and tile in (32, 64, 128, 256, 512, 1024)
                    and border <= 32 and n_off == layers):
                continue
            try:
                return _parse_vt(data, p, fmt, info)
            except (ValueError, struct.error, IndexError):
                continue
        return None
    return None


def _parse_vt(data, p, fmt, info):
    layers, _wb, _hb, tile, border, n_off = struct.unpack_from("<6I", data, p)
    p += 24
    layer_sizes = list(struct.unpack_from(f"<{n_off}I", data, p))
    p += 4 * n_off
    n_mips, width, height = struct.unpack_from("<3I", data, p)
    p += 12
    if not (1 <= n_mips <= 20 and 0 < width <= 1 << 20 and 0 < height <= 1 << 20):
        raise ValueError("bad VT header")

    def u32_array():
        nonlocal p
        n, = struct.unpack_from("<i", data, p)
        if not 0 <= n <= 1 << 20:
            raise ValueError("bad array")
        values = list(struct.unpack_from(f"<{n}I", data, p + 4))
        p += 4 + 4 * n
        return values

    chunk_per_mip = u32_array()
    base_per_mip = u32_array()
    n_tod, = struct.unpack_from("<i", data, p)
    p += 4
    tile_offsets = []
    for _ in range(n_tod):
        tw, th, _max = struct.unpack_from("<3I", data, p)
        p += 12
        tile_offsets.append((tw, th, u32_array(), u32_array()))
    for _ in range(3):  # TileIndexPerChunk, TileIndexPerMip, TileOffsetInChunk (UE4 style, usually empty)
        u32_array()
    layer_formats = []
    for _ in range(layers):
        s, p = _fstring(data, p)
        layer_formats.append(s)
    if not layer_formats[0].startswith("PF_"):
        raise ValueError("bad VT layer format")
    chunks = None
    # UE5.x may store per-layer fallback colors before the chunks; the per-layer codec entry is
    # u8 codec + u16 payload offset (older) or + u32 (newer). Keep the variant whose sizes add up.
    for colors, codec_size in ((True, 5), (True, 3), (False, 5), (False, 3)):
        q = p + (16 * layers if colors else 0)
        try:
            n_chunks, = struct.unpack_from("<i", data, q)
            if not 1 <= n_chunks <= 4096:
                continue
            q += 4
            found = []
            for _ in range(n_chunks):
                q += 20  # hash
                size, payload = struct.unpack_from("<II", data, q)
                q += 8
                codecs = []
                for _layer in range(layers):
                    codecs.append(data[q])
                    q += codec_size
                flags, bsize, boffset, inline, q = _read_bulk(data, q, info)
                if bsize != size:
                    raise ValueError("chunk size mismatch")
                found.append((size, payload, codecs, flags, bsize, boffset, inline))
            chunks = found
            break
        except (ValueError, struct.error, IndexError):
            continue
    if chunks is None:
        raise ValueError("bad VT chunks")
    return {"format": layer_formats[0], "tile": tile, "border": border, "tile_bytes": layer_sizes[0],
            "tile_data_size": layer_sizes[-1], "width": width, "height": height, "mips": n_mips,
            "chunk_per_mip": chunk_per_mip, "base_per_mip": base_per_mip, "tile_offsets": tile_offsets,
            "chunks": chunks}


def virtual_texture_image(vt, read_bulk):
    fmt, tile, border = vt["format"], vt["tile"], vt["border"]
    full = tile + 2 * border
    for mip in range(vt["mips"]):
        w, h = max(1, vt["width"] >> mip), max(1, vt["height"] >> mip)
        if w * h <= MAX_VT_PIXELS:
            break
    tw, th, addresses, offsets = vt["tile_offsets"][mip]
    chunk = vt["chunks"][vt["chunk_per_mip"][mip]]
    size, _payload, codecs, flags, bsize, boffset, inline = chunk
    raw = inline if inline is not None else read_bulk(flags, boffset, bsize)
    if codecs[0] == VT_ZIPPED_GPU:
        import zlib
        raw = raw[:_payload] + zlib.decompress(raw[_payload:])
    elif codecs[0] != VT_RAW_GPU:
        raise ValueError(f"Virtual texture codec {codecs[0]} isn't supported")
    from PIL import Image
    canvas = Image.new("RGBA", (tw * tile, th * tile))
    base = vt["base_per_mip"][mip]
    for ty in range(th):
        for tx in range(tw):
            address = _morton(tx) | (_morton(ty) << 1)
            block = max(i for i, a in enumerate(addresses) if a <= address) if addresses else 0
            if not offsets or offsets[block] == 0xFFFFFFFF:
                continue
            index = offsets[block] + address - addresses[block]
            at = base + index * vt["tile_data_size"]
            tile_img = decode_pixels(fmt, raw[at:at + vt["tile_bytes"]], full, full)
            canvas.paste(tile_img.crop((border, border, border + tile, border + tile)).convert("RGBA"),
                         (tx * tile, ty * tile))
    return canvas.crop((0, 0, w, h))


def texture_image(data, info, read_bulk):
    """PIL image of a texture. read_bulk(flags, offset, size) -> bytes from .ubulk/.uptnl (or raises)."""
    found = find_texture(data, info)
    if found is None:
        vt = find_virtual_texture(data, info)
        if vt is not None:
            return virtual_texture_image(vt, read_bulk)
        raise ValueError("No texture data found in this asset (it may be a render target or stripped).")
    fmt, _w, _h, mips, _cube = found
    errors = []
    for mip in mips:  # largest first
        if not mip.size or mip.flags & BULK_UNUSED:
            continue
        try:
            if mip.inline is not None:
                pixels = mip.inline
            else:
                pixels = read_bulk(mip.flags, mip.offset, mip.size)
            need = mip_bytes(fmt, mip.w, mip.h)
            if len(pixels) < need:
                raise ValueError(f"mip {mip.w}x{mip.h} has {len(pixels)} of {need} bytes")
            return decode_pixels(fmt, pixels[:need], mip.w, mip.h)
        except Exception as e:
            errors.append(f"{mip.w}x{mip.h}: {e}")
    raise ValueError("Couldn't read any mip of this texture (" + "; ".join(errors[:3]) + ")"
                     if errors else "This texture has no pixel data in the game files.")


def texture_size(data, info=None):
    found = find_texture(data, info)
    if found:
        return found[0], found[1], found[2]
    vt = find_virtual_texture(data, info)
    return (vt["format"] + " (virtual)", vt["width"], vt["height"]) if vt else None


# --------------------------------------------------------------------------- meshes

POSITIONS_RE = re.compile(rb"(?s)\x0c\x00\x00\x00(....)\x0c\x00\x00\x00\1")


def _packed_normals(raw, count, high_precision):
    if high_precision:
        t = np.frombuffer(raw, "<i2", count * 8).reshape(count, 8).astype(np.float32) / 32767.0
        return t[:, 4:7]
    t = np.frombuffer(raw, np.uint8, count * 8).reshape(count, 8).astype(np.float32)
    return t[:, 4:7] / 127.5 - 1.0


def find_mesh(buffers):
    """Look through byte buffers (export data, bulk data) for the most detailed vertex buffer."""
    best = None
    for buf in buffers:
        for m in POSITIONS_RE.finditer(buf):
            n, = struct.unpack("<i", m.group(1))
            start = m.end()
            if not (3 <= n <= 20_000_000) or start + n * 12 > len(buf):
                continue
            if best is None or n > best[1]:
                best = (buf, n, start)
    return best


def mesh_data(buffers, name=""):
    found = find_mesh(buffers)
    if found is None:
        raise ValueError("No mesh vertex data found in this asset (Nanite-only meshes and some "
                         "engine versions aren't supported).")
    buf, n, start = found
    points = np.frombuffer(buf, "<f4", n * 3, start).reshape(n, 3).astype(np.float32)
    pos = start + n * 12
    normals = uv = None
    # FStaticMeshVertexBuffer: strip flags, NumTexCoords, NumVertices, full precision UVs, high precision tangents
    for skip in (2, 0, 4):
        try:
            n_tex, n_verts, full_uv, high_tan = struct.unpack_from("<4I", buf, pos + skip)
        except struct.error:
            break
        if n_verts == n and 1 <= n_tex <= 8 and full_uv in (0, 1) and high_tan in (0, 1):
            p = pos + skip + 16
            elem, count = struct.unpack_from("<ii", buf, p)
            p += 8
            if count == n and elem in (8, 16):
                normals = _packed_normals(buf[p:p + n * elem], n, elem == 16)
                p += n * elem
                elem, count = struct.unpack_from("<ii", buf, p)
                p += 8
                if count in (n * n_tex, n) and elem in (4, 8, 4 * n_tex, 8 * n_tex):
                    per = (8 if full_uv else 4)
                    dtype = "<f4" if full_uv else "<f2"
                    raw = np.frombuffer(buf, dtype, n * n_tex * 2, p).reshape(n, n_tex, 2).astype(np.float32)
                    uv = raw[:, 0, :].copy()
                    p += n * n_tex * per
                pos = p
            break
    tris = _find_indices(buffers, buf, pos, start, n)
    if tris is None:
        raise ValueError("Found the vertices but not the triangle list of this mesh.")
    # Unreal: left-handed, Z up, X forward, centimeters -> right-handed Y up.
    p = np.stack([points[:, 1], points[:, 2], -points[:, 0]], axis=1)
    nrm = None
    if normals is not None:
        nrm = np.stack([normals[:, 1], normals[:, 2], -normals[:, 0]], axis=1)
    tris = orient_to_normals(p, tris, nrm) if nrm is not None else tris[:, ::-1]
    uvs = None
    if uv is not None:
        uv[:, 1] = 1.0 - uv[:, 1]
        uvs = {"UV0": uv}
    return MeshData(p, [tris], normals=nrm, uvs=uvs, name=name)


def _indices_at(buf, pos, n):
    """Try an index buffer (bulk array) at pos: returns (M, 3) array or None."""
    try:
        elem, count = struct.unpack_from("<ii", buf, pos)
    except struct.error:
        return None
    if elem not in (1, 2, 4) or count <= 0 or pos + 8 + elem * count > len(buf):
        return None
    raw = buf[pos + 8:pos + 8 + elem * count]
    for width in ((2, 4) if elem == 1 else (elem,)):
        if len(raw) % width:
            continue
        idx = np.frombuffer(raw, "<u2" if width == 2 else "<u4")
        if len(idx) >= 3 and len(idx) % 3 == 0 and int(idx.max()) < n and int(idx.max()) >= n // 2:
            return idx.astype(np.int64).reshape(-1, 3)
    return None


def _find_indices(buffers, buf, after, before, n):
    """Index buffer of the vertex buffer with n vertices.

    Static meshes keep it after the vertex buffers, skeletal meshes before them, and other LODs'
    index buffers are nearby too - so prefer a buffer that uses the last vertex (max index n-1),
    then the biggest one. Near the vertex data first; other buffers only if nothing fits.
    """
    pattern = re.compile(rb"[\x01\x02\x04]\x00\x00\x00")

    def near():
        window_end = after + 64 * 1024 + n * 64  # color / extra vertex buffers can come first
        for m in pattern.finditer(buf, after, window_end):
            yield buf, m.start()
        lo = max(0, before - 12 - n * 64 - 64 * 1024)
        for m in pattern.finditer(buf, lo, before):
            yield buf, m.start()

    def elsewhere():
        for other in buffers:
            if other is not buf:
                for m in pattern.finditer(other):
                    yield other, m.start()

    for candidates in (near(), elsewhere()):
        best, best_score = None, None
        for b, pos in candidates:
            tris = _indices_at(b, pos, n)
            if tris is None:
                continue
            score = (int(tris.max()) == n - 1, len(tris))
            if best is None or score > best_score:
                best, best_score = tris, score
        if best is not None:
            return best
    return None
