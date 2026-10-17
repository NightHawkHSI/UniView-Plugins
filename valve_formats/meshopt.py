"""meshoptimizer vertex/index buffer decoders (used by Source 2 and other modern engines).

Ported from meshoptimizer (MIT, (c) Arseny Kapoulkine) via ValveResourceFormat's C# port
(MIT, (c) ValveResourceFormat Contributors). Byte groups are decoded in Python; the per-channel
delta decoding is vectorized with numpy.
"""

import numpy as np

VERTEX_HEADER = 0xA0
INDEX_HEADER = 0xE0
BLOCK_BYTES = 8192
BLOCK_MAX = 256
GROUP = 16
BITS_V0 = (0, 2, 4, 8)
BITS_V1 = (0, 1, 2, 4, 8)


def _block_size(vertex_size):
    return min((BLOCK_BYTES // vertex_size) & ~(GROUP - 1), BLOCK_MAX)


def _decode_group(data, pos, out, out_pos, bits, reverse_bits):
    """Decode one 16-byte group; returns the new data position."""
    if bits == 0:
        out[out_pos:out_pos + GROUP] = b"\0" * GROUP
        return pos
    if bits == 8:
        out[out_pos:out_pos + GROUP] = data[pos:pos + GROUP]
        return pos + GROUP
    header_len = bits * 2  # 16 values * bits / 8
    extra = pos + header_len
    mask = (1 << bits) - 1
    per_byte = 8 // bits
    k = out_pos
    for h in range(header_len):
        b = data[pos + h]
        for i in range(per_byte):
            if reverse_bits:
                enc = (b >> i) & 1  # 1-bit groups are stored least significant bit first
            else:
                enc = (b >> (8 - bits * (i + 1))) & mask
            if enc == mask:
                out[k] = data[extra]
                extra += 1
            else:
                out[k] = enc
            k += 1
    return extra


def _decode_bytes(data, pos, out, count, bits_table):
    """Decode count (multiple of 16) bytes into out; returns new position."""
    groups = count // GROUP
    header_size = (groups + 3) // 4
    header = data[pos:pos + header_size]
    pos += header_size
    for g in range(groups):
        bitsk = (header[g // 4] >> ((g % 4) * 2)) & 3
        bits = bits_table[bitsk]
        pos = _decode_group(data, pos, out, g * GROUP, bits, bits == 1)
    return pos


def _unzigzag(v, dtype):
    v = v.astype(dtype)
    return (np.zeros_like(v) - (v & 1)) ^ (v >> 1)


def decode_vertex_buffer(count, size, data):
    """meshopt-compressed vertex buffer -> bytes (count * size)."""
    if size <= 0 or size > 256 or size % 4:
        raise ValueError("Vertex size must be a multiple of 4 up to 256")
    if not data or (data[0] & 0xF0) != VERTEX_HEADER:
        raise ValueError("Not a meshopt vertex buffer")
    version = data[0] & 0x0F
    if version > 1:
        raise ValueError(f"Unsupported meshopt vertex version {version}")
    data = bytes(data)
    tail = size + (0 if version == 0 else size // 4)
    last = bytearray(data[len(data) - tail:len(data) - tail + size])
    channels = data[len(data) - tail + size:len(data) - tail + size + size // 4] if version else None
    out = np.empty((count, size), np.uint8)
    pos = 1
    block = _block_size(size)
    offset = 0
    scratch = bytearray(BLOCK_MAX * 4)
    while offset < count:
        n = min(block, count - offset)
        aligned = (n + GROUP - 1) // GROUP * GROUP
        control_size = 0 if version == 0 else size // 4
        control = data[pos:pos + control_size]
        pos += control_size
        for k in range(0, size, 4):
            ctrl_byte = 0 if version == 0 else control[k // 4]
            streams = []
            for j in range(4):
                ctrl = (ctrl_byte >> (j * 2)) & 3
                if ctrl == 3:
                    streams.append(np.frombuffer(data, np.uint8, n, pos))
                    pos += n
                elif ctrl == 2:
                    streams.append(np.zeros(n, np.uint8))
                else:
                    buf = bytearray(aligned)
                    table = BITS_V0 if version == 0 else BITS_V1[ctrl:]
                    pos = _decode_bytes(data, pos, buf, aligned, table)
                    streams.append(np.frombuffer(bytes(buf), np.uint8, n))
            channel = 0 if version == 0 else channels[k // 4]
            kind = channel & 3
            if kind == 0:  # 1-byte deltas
                for j in range(4):
                    d = _unzigzag(streams[j], np.uint8)
                    col = (np.cumsum(d.astype(np.uint32)) + last[k + j]) & 0xFF
                    out[offset:offset + n, k + j] = col
            elif kind == 1:  # 2-byte deltas
                for j in (0, 2):
                    v = streams[j].astype(np.uint16) | (streams[j + 1].astype(np.uint16) << 8)
                    d = _unzigzag(v, np.uint16).astype(np.uint32)
                    p = last[k + j] | (last[k + j + 1] << 8)
                    col = (np.cumsum(d) + p) & 0xFFFF
                    out[offset:offset + n, k + j] = col & 0xFF
                    out[offset:offset + n, k + j + 1] = col >> 8
            elif kind == 2:  # 4-byte xor with rotation
                rot = (32 - (channel >> 4)) & 31
                v = (streams[0].astype(np.uint32) | (streams[1].astype(np.uint32) << 8)
                     | (streams[2].astype(np.uint32) << 16) | (streams[3].astype(np.uint32) << 24))
                if rot:
                    v = ((v << np.uint32(rot)) | (v >> np.uint32(32 - rot))) & np.uint32(0xFFFFFFFF)
                p = np.uint32(int.from_bytes(bytes(last[k:k + 4]), "little"))
                col = np.bitwise_xor.accumulate(v) ^ p
                for j in range(4):
                    out[offset:offset + n, k + j] = (col >> np.uint32(8 * j)) & 0xFF
            else:
                raise ValueError("Invalid meshopt channel")
        last[:] = out[offset + n - 1].tobytes()
        offset += n
    del scratch
    return out.tobytes()


def _vbyte(data, pos):
    lead = data[pos]
    pos += 1
    if lead < 128:
        return lead, pos
    result = lead & 127
    shift = 7
    for _ in range(4):
        group = data[pos]
        pos += 1
        result |= (group & 127) << shift
        shift += 7
        if group < 128:
            break
    return result, pos


def decode_index_buffer(count, size, data):
    """meshopt-compressed triangle index buffer -> numpy uint32 array (count,)."""
    if count % 3:
        raise ValueError("Index count must be a multiple of 3")
    data = bytes(data)
    if (data[0] & 0xF0) != INDEX_HEADER:
        raise ValueError("Not a meshopt index buffer")
    version = data[0] & 0x0F
    if version > 1:
        raise ValueError(f"Unsupported meshopt index version {version}")
    tris = count // 3
    code = data[1:1 + tris]
    body = data[1 + tris:]
    safe_end = len(body) - 16
    codeaux_table = body[safe_end:]
    edge = [(0, 0)] * 16
    verts = [0] * 16
    eo = vo = 0
    nxt = last = 0
    pos = 0
    fecmax = 13 if version >= 1 else 15
    out = [0] * count
    o = 0
    M = 0xFFFFFFFF
    for codetri in code:
        if codetri < 0xF0:
            fe = codetri >> 4
            a, b = edge[(eo - 1 - fe) & 15]
            fec = codetri & 15
            if fec < fecmax:
                if fec == 0:
                    c = nxt
                    nxt += 1
                    verts[vo] = c
                    vo = (vo + 1) & 15
                else:
                    c = verts[(vo - 1 - fec) & 15]
                    verts[vo] = c
            else:
                if fec != 15:
                    c = (last + fec * 2 - 27) & M
                else:
                    v, pos = _vbyte(body, pos)
                    c = (last + ((v >> 1) ^ -(v & 1))) & M
                last = c
                verts[vo] = c
                vo = (vo + 1) & 15
            edge[eo] = (c, b)
            eo = (eo + 1) & 15
            edge[eo] = (a, c)
            eo = (eo + 1) & 15
            out[o], out[o + 1], out[o + 2] = a, b, c
        elif codetri < 0xFE:
            codeaux = codeaux_table[codetri & 15]
            feb, fec = codeaux >> 4, codeaux & 15
            a = nxt
            nxt += 1
            if feb == 0:
                b = nxt
                nxt += 1
            else:
                b = verts[(vo - feb) & 15]
            if fec == 0:
                c = nxt
                nxt += 1
            else:
                c = verts[(vo - fec) & 15]
            out[o], out[o + 1], out[o + 2] = a, b, c
            verts[vo] = a
            vo = (vo + 1) & 15
            verts[vo] = b
            vo = (vo + (1 if feb == 0 else 0)) & 15
            verts[vo] = c
            vo = (vo + (1 if fec == 0 else 0)) & 15
            edge[eo] = (b, a)
            eo = (eo + 1) & 15
            edge[eo] = (c, b)
            eo = (eo + 1) & 15
            edge[eo] = (a, c)
            eo = (eo + 1) & 15
        else:
            codeaux = body[pos]
            pos += 1
            fea = 0 if codetri == 0xFE else 15
            feb, fec = codeaux >> 4, codeaux & 15
            if codeaux == 0:
                nxt = 0
            if fea == 0:
                a = nxt
                nxt += 1
            else:
                a = 0
            if feb == 0:
                b = nxt
                nxt += 1
            else:
                b = verts[(vo - feb) & 15]
            if fec == 0:
                c = nxt
                nxt += 1
            else:
                c = verts[(vo - fec) & 15]
            if fea == 15:
                v, pos = _vbyte(body, pos)
                a = last = (last + ((v >> 1) ^ -(v & 1))) & M
            if feb == 15:
                v, pos = _vbyte(body, pos)
                b = last = (last + ((v >> 1) ^ -(v & 1))) & M
            if fec == 15:
                v, pos = _vbyte(body, pos)
                c = last = (last + ((v >> 1) ^ -(v & 1))) & M
            out[o], out[o + 1], out[o + 2] = a, b, c
            verts[vo] = a
            vo = (vo + 1) & 15
            verts[vo] = b
            vo = (vo + (1 if feb in (0, 15) else 0)) & 15
            verts[vo] = c
            vo = (vo + (1 if fec in (0, 15) else 0)) & 15
            edge[eo] = (b, a)
            eo = (eo + 1) & 15
            edge[eo] = (c, b)
            eo = (eo + 1) & 15
            edge[eo] = (a, c)
            eo = (eo + 1) & 15
        o += 3
    if pos != safe_end:
        raise ValueError("Malformed meshopt index buffer")
    return np.array(out, dtype=np.uint32)
