"""Binary KeyValues3 (KV3) reader, the data format inside Source 2 compiled resources.

Ported from ValveResourceFormat's BinaryKV3.cs (MIT License, (c) ValveResourceFormat
Contributors). Values come back as plain Python: dict, list, str, int, float, bool, bytes, None.
"""

import struct

MAGIC_VKV3 = 0x03564B56
FRAME_SIZE = 16384
TRAILER = 0xFFEEDD00

NULL, BOOLEAN, INT64, UINT64, DOUBLE, STRING, BLOB, ARRAY, OBJECT, ARRAY_TYPED = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
INT32, UINT32, TRUE, FALSE, INT64_ZERO, INT64_ONE, DOUBLE_ZERO, DOUBLE_ONE, FLOAT = 11, 12, 13, 14, 15, 16, 17, 18, 19
INT16, UINT16, UNKNOWN_22, INT32_AS_BYTE, ARRAY_BYTE_LENGTH, ARRAY_AUX = 20, 21, 22, 23, 24, 25


def _lz4(data, size, dictionary=b""):
    import lz4.block
    if dictionary:
        return lz4.block.decompress(data, uncompressed_size=size, dict=dictionary)
    return lz4.block.decompress(data, uncompressed_size=size)


def _zstd(data, size):
    try:
        import zstandard
    except ImportError:
        try:
            from compression import zstd  # Python 3.14+
            return zstd.decompress(data)
        except ImportError:
            raise RuntimeError("This file uses zstd compression - install it with: py -m pip install zstandard")
    return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)


class _Buf:
    """A byte region read from the front (one of the 1/2/4/8-byte value streams)."""

    __slots__ = ("data", "pos")

    def __init__(self, data=b""):
        self.data, self.pos = data, 0

    def take(self, fmt, size):
        value = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def byte(self):
        value = self.data[self.pos]
        self.pos += 1
        return value


class _Buffers:
    def __init__(self):
        self.b1, self.b2, self.b4, self.b8 = _Buf(), _Buf(), _Buf(), _Buf()


def _align(n, a):
    return (n + a - 1) // a * a


def _split(raw, offset, c1, c2, c4, c8, align_empty8):
    bufs = _Buffers()
    if c1:
        bufs.b1 = _Buf(raw[offset:offset + c1])
        offset += c1
    if c2:
        offset = _align(offset, 2)
        bufs.b2 = _Buf(raw[offset:offset + c2 * 2])
        offset += c2 * 2
    if c4:
        offset = _align(offset, 4)
        bufs.b4 = _Buf(raw[offset:offset + c4 * 4])
        offset += c4 * 4
    if c8:
        offset = _align(offset, 8)
        bufs.b8 = _Buf(raw[offset:offset + c8 * 8])
        offset += c8 * 8
    elif align_empty8:
        offset = _align(offset, 8)
    return bufs, offset


def _cstrings(raw, offset, count):
    out = []
    for _ in range(count):
        end = raw.index(b"\0", offset)
        out.append(raw[offset:end].decode("utf-8", "replace"))
        offset = end + 1
    return out, offset


class _Ctx:
    pass


def read_kv3(data):
    """Parse a binary KV3 block (bytes starting with its magic)."""
    magic, = struct.unpack_from("<I", data, 0)
    if magic == MAGIC_VKV3:
        raise ValueError("Legacy VKV3 blocks aren't supported")
    version = magic & 0xFF
    if magic & 0xFFFFFF00 != 0x4B563300 or not 1 <= version <= 5:
        raise ValueError("Not a KV3 block")
    pos = 4 + 16
    method, = struct.unpack_from("<I", data, pos)
    pos += 4
    frame_size = 0
    c1 = c4 = c8 = n_types = 0
    size_total = csize_total = n_blocks = blob_bytes = 0
    if version == 1:
        c1, c4, c8, size_total = struct.unpack_from("<4i", data, pos)
        pos += 16
        csize_total = len(data) - pos
    else:
        _dict_id, frame_size = struct.unpack_from("<HH", data, pos)
        pos += 4
        c1, c4, c8, n_types = struct.unpack_from("<4i", data, pos)
        pos += 16
        pos += 4  # object / array counts (u16 each)
        size_total, csize_total, n_blocks, blob_bytes = struct.unpack_from("<4i", data, pos)
        pos += 16
    c2 = 0
    if version >= 4:
        c2, _block_sizes_bytes = struct.unpack_from("<2i", data, pos)
        pos += 8
    if version >= 5:
        (size1, csize1, size2, csize2, c1_2, c2_2, c4_2, c8_2, _u13, n_obj2, _n_arr2, _u16) = \
            struct.unpack_from("<12i", data, pos)
        pos += 48
    else:
        size1, csize1 = size_total, csize_total

    def unpack(length, csize, extra=0):
        nonlocal pos
        if method == 0:
            out = data[pos:pos + length]
            pos += length
            return out
        chunk = data[pos:pos + csize]
        pos += csize
        if method == 1:
            return _lz4(chunk, length)
        if method == 2:
            return _zstd(chunk, length + extra)
        raise ValueError(f"Unknown KV3 compression {method}")

    ctx = _Ctx()
    ctx.version = version
    blob_sizes = None
    raw1 = unpack(size1, csize1 if (method or version < 5) else 0,
                  blob_bytes if (method == 2 and version < 5) else 0)
    bufs1, offset = _split(raw1, 0, c1, c2, c4, c8, version < 5)
    n_strings = bufs1.b4.take("<i", 4)
    if version >= 5:
        ctx.aux = bufs1
        # strings come first in the 1-byte stream; auxiliary array bytes follow them
        ctx.strings, bufs1.b1.pos = _cstrings(bufs1.b1.data, 0, n_strings)
    else:
        ctx.buf = bufs1
        start = offset
        ctx.strings, offset = _cstrings(raw1, offset, n_strings)
        types_len = (size_total - offset - 4) if version == 1 else (n_types - offset + start)
        ctx.types = _Buf(raw1[offset:offset + types_len])
        offset += types_len
        if n_blocks:
            blob_sizes = raw1[offset:]
    if version >= 5:
        raw2 = unpack(size2, csize2)
        bufs2 = _Buffers()
        end = n_obj2 * 4
        ctx.object_lengths = _Buf(raw2[:end])
        bufs2, offset = _split(raw2, end, c1_2, c2_2, c4_2, c8_2, False)
        ctx.buf = bufs2
        ctx.types = _Buf(raw2[offset:offset + n_types])
        offset += n_types
        if n_blocks:
            blob_sizes = raw2[offset:]
    ctx.blob_lengths = _Buf()
    ctx.blobs = _Buf()
    if n_blocks:
        lengths = blob_sizes[:n_blocks * 4]
        ctx.blob_lengths = _Buf(lengths)
        rest = blob_sizes[n_blocks * 4 + 4:]  # skip trailer
        if method == 0:
            ctx.blobs = _Buf(data[pos:pos + blob_bytes])
            pos += blob_bytes
        elif method == 1:
            out = bytearray()
            p = 0
            while p + 2 <= len(rest) and len(out) < blob_bytes:
                clen, = struct.unpack_from("<H", rest, p)
                p += 2
                want = min(frame_size or FRAME_SIZE, blob_bytes - len(out))
                out += _lz4(data[pos:pos + clen], want, bytes(out[-65536:]))
                pos += clen
            ctx.blobs = _Buf(bytes(out))
        elif method == 2:
            if version >= 5:
                size = csize_total - csize1 - csize2
                ctx.blobs = _Buf(_zstd(data[pos:pos + size], blob_bytes))
                pos += size
            else:
                ctx.blobs = _Buf(raw1[size1:size1 + blob_bytes])
    kind, _flag = _read_type(ctx)
    return _read_value(ctx, kind)


def _read_type(ctx):
    t = ctx.types.byte()
    flag = 0
    if ctx.version >= 3:
        if t & 0x80:
            t &= 0x3F
            flag = ctx.types.byte()
    elif t & 0x80:
        t &= 0x7F
        flag = ctx.types.byte()
    return t, flag


def _read_value(ctx, t):
    b = ctx.buf
    if t == NULL:
        return None
    if t == TRUE:
        return True
    if t == FALSE:
        return False
    if t == INT64_ZERO:
        return 0
    if t == INT64_ONE:
        return 1
    if t == DOUBLE_ZERO:
        return 0.0
    if t == DOUBLE_ONE:
        return 1.0
    if t == BOOLEAN:
        return b.b1.byte() == 1
    if t in (INT32_AS_BYTE, UNKNOWN_22):
        return b.b1.byte()
    if t == INT16:
        return b.b2.take("<h", 2)
    if t == UINT16:
        return b.b2.take("<H", 2)
    if t == INT32:
        return b.b4.take("<i", 4)
    if t == UINT32:
        return b.b4.take("<I", 4)
    if t == FLOAT:
        return b.b4.take("<f", 4)
    if t == INT64:
        return b.b8.take("<q", 8)
    if t == UINT64:
        return b.b8.take("<Q", 8)
    if t == DOUBLE:
        return b.b8.take("<d", 8)
    if t == STRING:
        i = b.b4.take("<i", 4)
        return "" if i == -1 else ctx.strings[i]
    if t == BLOB:
        if ctx.version < 2:
            n = b.b4.take("<i", 4)
            out = b.b1.data[b.b1.pos:b.b1.pos + n]
            b.b1.pos += n
            return bytes(out)
        n = ctx.blob_lengths.take("<i", 4)
        out = ctx.blobs.data[ctx.blobs.pos:ctx.blobs.pos + n]
        ctx.blobs.pos += n
        return bytes(out)
    if t == ARRAY:
        n = b.b4.take("<i", 4)
        out = []
        for _ in range(n):
            sub, _f = _read_type(ctx)
            out.append(_read_value(ctx, sub))
        return out
    if t in (ARRAY_TYPED, ARRAY_BYTE_LENGTH):
        n = b.b1.byte() if t == ARRAY_BYTE_LENGTH else b.b4.take("<i", 4)
        sub, _f = _read_type(ctx)
        return [_read_value(ctx, sub) for _ in range(n)]
    if t == ARRAY_AUX:
        n = b.b1.byte()
        sub, _f = _read_type(ctx)
        ctx.aux, ctx.buf = ctx.buf, ctx.aux
        try:
            return [_read_value(ctx, sub) for _ in range(n)]
        finally:
            ctx.aux, ctx.buf = ctx.buf, ctx.aux
    if t == OBJECT:
        n = ctx.object_lengths.take("<i", 4) if ctx.version >= 5 else b.b4.take("<i", 4)
        out = {}
        for _ in range(n):
            sub, _f = _read_type(ctx)
            i = ctx.buf.b4.take("<i", 4)
            out["" if i == -1 else ctx.strings[i]] = _read_value(ctx, sub)
        return out
    raise ValueError(f"Unknown KV3 type {t}")
