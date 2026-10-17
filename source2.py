"""Source 2 engine plugin (CS2, Dota 2, Half-Life: Alyx, Deadlock, s&box ...).

Reads the game's VPK packs: models (.vmdl_c, incl. meshoptimizer-compressed buffers) with their
materials' textures, compiled textures (.vtex_c) and text files; everything else exports as-is.
Model/KV3/meshopt decoding follows ValveResourceFormat (MIT).
"""

import logging
import os
import re
import struct

import numpy as np
from PIL import Image

from engines.sdk import (
    Asset, EnginePlugin, GameSession, Kind, kind_for_extension, listdir_lower, pil_image_from_bytes,
)
from valve_formats.vpk import VPK, VirtualFS, vpk_size

log = logging.getLogger("viewer.source2")

SKIP_SUFFIXES = ("_brazilian", "_bulgarian", "_czech", "_danish", "_dutch", "_english", "_finnish",
                 "_french", "_german", "_greek", "_hungarian", "_indonesian", "_italian", "_japanese",
                 "_koreana", "_korean", "_latam", "_norwegian", "_polish", "_portuguese", "_romanian",
                 "_russian", "_schinese", "_spanish", "_swedish", "_tchinese", "_thai", "_turkish",
                 "_ukrainian", "_vietnamese", "_lv", "_addons", "_community_addons", "_imported")
LAST_MODS = ("core",)


def game_root(path):
    """The folder holding the mod folders (…/game), or None."""
    if os.path.isfile(os.path.join(path, "gameinfo.gi")):
        return os.path.dirname(path)
    game = os.path.join(path, "game")
    if os.path.isdir(os.path.join(game, "bin")):
        return game
    return None


def mod_dirs(path):
    if os.path.isfile(os.path.join(path, "gameinfo.gi")):
        return [path]
    root = game_root(path)
    if root is None:
        return []
    mods = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub) or name.lower().endswith(SKIP_SUFFIXES) or name.lower() == "bin":
            continue
        files = listdir_lower(sub)
        if "gameinfo.gi" in files or any(f.endswith("_dir.vpk") for f in files):
            mods.append(sub)
    return sorted(mods, key=lambda d: os.path.basename(d).lower() in LAST_MODS)


def dir_vpks(mod):
    try:
        return [os.path.join(mod, n) for n in sorted(os.listdir(mod)) if n.lower().endswith("_dir.vpk")]
    except OSError:
        return []


# --------------------------------------------------------------------------- compiled resources

def resource_blocks(data):
    """{block type: (offset, size)} of a Source 2 compiled resource (…_c file)."""
    if len(data) < 16:
        raise ValueError("File too small to be a Source 2 resource")
    _size, header_version, _type_version, block_offset, block_count = struct.unpack_from("<IHHII", data, 0)
    if header_version != 12:
        raise ValueError(f"Unknown resource header version {header_version}")
    blocks: dict[str, tuple[int, int]] = {}
    pos = 8 + block_offset
    for _ in range(block_count):
        kind = data[pos:pos + 4].decode("ascii", "replace")
        rel, size = struct.unpack_from("<II", data, pos + 4)
        blocks.setdefault(kind, (pos + 4 + rel, size))
        pos += 12
    return blocks


def resource_block_list(data):
    """[(block type, offset, size)] in file order (models refer to blocks by this index)."""
    _size, _hv, _tv, block_offset, block_count = struct.unpack_from("<IHHII", data, 0)
    out = []
    pos = 8 + block_offset
    for _ in range(block_count):
        kind = data[pos:pos + 4].decode("ascii", "replace")
        rel, size = struct.unpack_from("<II", data, pos + 4)
        out.append((kind, pos + 4 + rel, size))
        pos += 12
    return out


def block_kv3(data, blocks, kind):
    from valve_formats.kv3 import read_kv3
    for k, off, size in blocks:
        if k == kind:
            return read_kv3(data[off:off + size])
    return None


# --------------------------------------------------------------------------- other resources as data

# Compiled resource types shown as readable data, with a friendly type name.
DATA_TYPES = {
    "vpcf_c": "Particle system", "vmat_c": "Material", "vdata_c": "Game data", "vsmart_c": "Smart prop",
    "vsndevts_c": "Sound events", "vsndstck_c": "Sound stack", "vnmclip_c": "Animation clip",
    "vnmgraph_c": "Animation graph", "vnmskel_c": "Skeleton", "vnmvar_c": "Animation variations",
    "vanmgrph_c": "Animation graph", "vseq_c": "Sequence", "vcd_c": "Choreography scene",
    "vrr_c": "Response rules", "vsnap_c": "Snapshot", "vpost_c": "Post-processing", "vpulse_c": "Pulse graph",
    "vmix_c": "Sound mix", "vsurf_c": "Surface properties", "vphys_c": "Physics", "vmdl_c": "Model",
    "vwrld_c": "World", "vwnod_c": "World node", "vents_c": "Entity lump", "vmap_c": "Map",
    "vxml_c": "UI layout", "vcss_c": "UI style sheet", "vjs_c": "UI script", "vts_c": "UI script (TypeScript)",
    "vrman_c": "Resource manifest", "vcdlist_c": "Choreography list", "vpdi_c": "Dynamic images",
    "vagrp_c": "Animation group", "vanim_c": "Animation", "vphysxml_c": "Physics",
}
PANORAMA_EXTS = {"vxml_c": "xml", "vcss_c": "css", "vjs_c": "js", "vts_c": "ts", "vsvg_c": "svg"}


def external_refs(data, blocks):
    """File names a resource refers to (RERL block): materials, textures, sounds..."""
    for kind, off, size in blocks:
        if kind != "RERL" or size < 8:
            continue
        rel, count = struct.unpack_from("<II", data, off)
        names, pos = [], off + rel
        for _ in range(min(count, 100000)):
            name_rel = struct.unpack_from("<I", data, pos + 8)[0]
            start = pos + 8 + name_rel
            end = data.find(b"\0", start)
            names.append(data[start:end if end >= 0 else start].decode("utf-8", "replace"))
            pos += 16
        return names
    return []


def panorama_text(data, blocks):
    """UI files (Panorama): the DATA block is a CRC32 and a name table, then the text itself."""
    block = next(((o, sz) for k, o, sz in blocks if k == "DATA"), None)
    if block is None:
        return None
    raw = data[block[0]:block[0] + block[1]]
    count = struct.unpack_from("<H", raw, 4)[0] if len(raw) >= 6 else 0
    pos = 6
    if count < 1000:
        for _ in range(count):  # (name, two uint32) entries
            end = raw.find(b"\0", pos)
            if end < 0:
                pos = 0
                break
            pos = end + 1 + 8
    else:
        pos = 0  # no header at all (scripts)
    text = raw[pos:].decode("utf-8", "replace")
    sample = text[:4000]
    printable = sum(c.isprintable() or c in "\r\n\t" for c in sample)
    return text if sample and printable >= 0.97 * len(sample) else None  # binary data isn't text


def resource_data(data):
    """(decoded content, export extension, referenced files) of a compiled resource.

    Content is the DATA block as a dict (KeyValues3), UI text, or the readable strings found in it."""
    blocks = resource_block_list(data)
    refs = external_refs(data, blocks)
    try:
        tree = block_kv3(data, blocks, "DATA")
    except RuntimeError:
        raise  # a missing decompressor: say so
    except Exception:
        tree = None
    if tree is not None:
        return tree, "json", refs
    for kind, _off, _size in blocks:  # e.g. UI layouts keep their content in a 'LaCo' KV3 block
        if kind not in ("DATA", "RED2", "RERL", "REDI", "NTRO"):
            try:
                tree = block_kv3(data, blocks, kind)
            except Exception:
                tree = None
            if tree:
                return tree, "json", refs
    text = panorama_text(data, blocks)
    if text and text.strip():
        return text, "txt", refs
    block = next(((o, sz) for k, o, sz in blocks if k == "DATA"), (0, len(data)))
    from engines.sdk import find_cstrings
    return find_cstrings(data[block[0]:block[0] + block[1]]), "json", refs


def _json_safe(value, depth=0):
    if depth > 200:
        return "..."
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
        return f"<{len(b)} bytes>" if len(b) > 64 else b.hex()
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), depth + 1)
    if isinstance(value, float) and value != value:
        return "NaN"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# --------------------------------------------------------------------------- models (.vmdl_c)

# DXGI formats used by vertex attributes: id -> (numpy dtype, components, scale)
DXGI = {2: ("<f4", 4, None), 6: ("<f4", 3, None), 10: ("<f2", 4, None), 16: ("<f4", 2, None),
        28: ("u1", 4, 255.0), 30: ("u1", 4, None), 34: ("<f2", 2, None), 35: ("<u2", 2, 65535.0),
        36: ("<u2", 2, None), 37: ("<i2", 2, 32767.0), 38: ("<i2", 2, None), 41: ("<f4", 1, None),
        42: ("<u4", 1, None)}


def _attribute(vdata, count, stride, field):
    fmt = field["format"]
    if fmt not in DXGI:
        return None
    dtype, comps, scale = DXGI[fmt]
    rec = np.dtype({"names": ["v"], "formats": [(dtype, (comps,))], "offsets": [field["offset"]],
                    "itemsize": stride})
    arr = np.frombuffer(vdata, rec, count)["v"].astype(np.float32 if scale or dtype.startswith("<f") else np.int64)
    if scale:
        arr = arr.astype(np.float32) / scale
    return arr.reshape(count, comps)


def _normals(vdata, count, stride, field):
    fmt = field["format"]
    if fmt == 6:
        return _attribute(vdata, count, stride, field)
    if fmt == 42:  # CS2-style packed tangent frame
        packed = _attribute(vdata, count, stride, field)[:, 0].astype(np.uint32)
        x = ((packed >> 12) & 0x3FF) / 1023.0 * 2 - 1
        y = ((packed >> 22) & 0x3FF) / 1023.0 * 2 - 1
        z = 1 - np.abs(x) - np.abs(y)
        neg = np.clip(-z, 0, 1)
        x = x + np.where(x >= 0, -neg, neg)
        y = y + np.where(y >= 0, -neg, neg)
        n = np.stack([x, y, z], axis=1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    if fmt == 28:  # older octahedral-ish 2-byte normal
        raw = np.frombuffer(vdata, np.uint8).reshape(-1, stride)[:count, field["offset"]:field["offset"] + 2]
        x, y = raw[:, 0].astype(np.float32) - 128, raw[:, 1].astype(np.float32) - 128
        zs, ts = (x < 0).astype(np.float32), (y < 0).astype(np.float32)
        zsign, tsign = -(2 * zs - 1), -(2 * ts - 1)
        x = x * zsign - zs - 64
        y = y * tsign - ts - 64
        xs, ys = (x < 0).astype(np.float32), (y < 0).astype(np.float32)
        xsign, ysign = -(2 * xs - 1), -(2 * ys - 1)
        x = (x * xsign - xs) / 63.0
        y = (y * ysign - ys) / 63.0
        z = 1 - x - y
        n = np.stack([x * xsign, y * ysign, z * zsign], axis=1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    return None


def _decode_buffer(raw, count, size, is_vertex, meshopt, zstd):
    if zstd:
        from valve_formats.kv3 import _zstd
        raw = _zstd(raw, count * size)
    if meshopt and len(raw) < count * size:
        from valve_formats import meshopt as mo
        if is_vertex:
            return mo.decode_vertex_buffer(count, size, raw)
        return mo.decode_index_buffer(count, size, raw).astype("<u4" if size == 4 else "<u2").tobytes()
    return bytes(raw[:count * size])


def _vbib_buffers(data, off):
    """Old MBUF/VBIB block -> (vertex buffers, index buffers) as dicts."""
    vb_off, vb_count, ib_off, ib_count = struct.unpack_from("<4I", data, off)

    def read(pos, is_vertex):
        count, size_flags = struct.unpack_from("<Ii", data, pos)
        size = size_flags & 0x3FFFFFF
        meshopt = (size_flags & 0x4000000) == 0
        zstd = (size_flags & 0x8000000) != 0
        ref_a = pos + 8
        attr_off, attr_count = struct.unpack_from("<II", data, ref_a)
        ref_b = ref_a + 8
        data_off, total = struct.unpack_from("<Ii", data, ref_b)
        fields = []
        p = ref_a + attr_off
        for _ in range(attr_count):
            name = data[p:p + 32].split(b"\0")[0].decode("ascii", "replace").upper()
            sem_index, fmt, offset = struct.unpack_from("<iII", data, p + 32)
            fields.append({"name": name, "index": sem_index, "format": fmt, "offset": offset})
            p += 56
        raw = data[ref_b + data_off:ref_b + data_off + total]
        return {"count": count, "size": size, "fields": fields,
                "data": _decode_buffer(raw, count, size, is_vertex, meshopt and len(raw) < count * size, zstd)}

    vbs = [read(off + vb_off + i * 24, True) for i in range(vb_count)]
    ibs = [read(off + 8 + ib_off + i * 24, False) for i in range(ib_count)]
    return vbs, ibs


def _kv_buffers(data, blocks, entries, is_vertex):
    out = []
    for e in entries or []:
        fields = []
        for f in e.get("m_inputLayoutFields") or []:
            name = f.get("m_pSemanticName", "")
            if isinstance(name, bytes):
                name = name.split(b"\0")[0].decode("ascii", "replace")
            fields.append({"name": str(name).upper(), "index": f.get("m_nSemanticIndex", 0),
                           "format": f.get("m_Format", 0), "offset": f.get("m_nOffset", 0)})
        count, size = e["m_nElementCount"], e["m_nElementSizeInBytes"]
        if "m_pData" in e:
            raw = e["m_pData"]
            meshopt, zstd = len(raw) != count * size, False
        else:
            _k, off, bsize = blocks[e["m_nBlockIndex"]]
            raw = data[off:off + bsize]
            meshopt = bool(e.get("m_bMeshoptCompressed"))
            zstd = bool(e.get("m_bCompressedZSTD"))
        out.append({"count": count, "size": size, "fields": fields,
                    "data": _decode_buffer(raw, count, size, is_vertex, meshopt, zstd)})
    return out


def model_parts(data):
    """Decode a .vmdl_c: [(mesh name, CRenderMesh dict, vertex buffers, index buffers)] at the best LOD."""
    blocks = resource_block_list(data)
    ctrl = block_kv3(data, blocks, "CTRL") or {}
    info = block_kv3(data, blocks, "DATA") or {}
    masks = info.get("m_refLODGroupMasks") or []
    combined = 0
    for m in masks:
        combined |= int(m)
    lowest = (combined & -combined).bit_length() - 1 if combined else 0
    parts = []
    for em in ctrl.get("embedded_meshes") or []:
        index = em.get("m_nMeshIndex", em.get("mesh_index", 0))
        if index < len(masks) and masks[index] and not (int(masks[index]) >> lowest) & 1:
            continue  # a lower level of detail
        data_block = em.get("m_nDataBlock", em.get("data_block"))
        _k, off, size = blocks[data_block]
        from valve_formats.kv3 import read_kv3
        mesh = read_kv3(data[off:off + size])
        if "vbib_block" in em:
            _k, voff, _vsize = blocks[em["vbib_block"]]
            vbs, ibs = _vbib_buffers(data, voff)
        else:
            vbs = _kv_buffers(data, blocks, em.get("m_vertexBuffers"), True)
            ibs = _kv_buffers(data, blocks, em.get("m_indexBuffers"), False)
        parts.append((em.get("m_Name", em.get("name", "")), mesh, vbs, ibs))
    return parts, info


def model_meshdata(data, name=""):
    """MeshData + material paths (one submesh per draw call)."""
    from engines.sdk import MeshData, orient_to_normals, z_up_to_y_up
    parts, _info = model_parts(data)
    points, normals, uvs, submeshes, slots, materials = [], [], [], [], [], []
    base = 0
    for _name, mesh, vbs, ibs in parts:
        vb_base = {}
        for i, vb in enumerate(vbs):
            fields = {(f["name"], f["index"]): f for f in vb["fields"]}
            pos = fields.get(("POSITION", 0))
            if pos is None:
                continue
            count, stride = vb["count"], vb["size"]
            p = _attribute(vb["data"], count, stride, pos)
            n = _normals(vb["data"], count, stride, fields[("NORMAL", 0)]) if ("NORMAL", 0) in fields else None
            uv_field = fields.get(("TEXCOORD", 0))
            uv = _attribute(vb["data"], count, stride, uv_field) if uv_field else None
            points.append(p[:, :3])
            normals.append(n if n is not None else np.zeros((count, 3), np.float32))
            uvs.append(uv[:, :2] if uv is not None and uv.shape[1] >= 2 else np.zeros((count, 2), np.float32))
            vb_base[i] = base
            base += count
        index_arrays = [np.frombuffer(ib["data"], "<u4" if ib["size"] == 4 else "<u2").astype(np.int64) for ib in ibs]
        for obj in mesh.get("m_sceneObjects") or []:
            for call in obj.get("m_drawCalls") or []:
                if "TRIANGLES" not in str(call.get("m_nPrimitiveType", "RENDER_PRIM_TRIANGLES")):
                    continue
                ib = (call.get("m_indexBuffer") or {}).get("m_hBuffer", 0)
                vb = ((call.get("m_vertexBuffers") or [{}])[0]).get("m_hBuffer", 0)
                if ib >= len(index_arrays) or vb not in vb_base:
                    continue
                start, n_idx = call.get("m_nStartIndex", 0), call.get("m_nIndexCount", 0)
                idx = index_arrays[ib][start:start + n_idx] + call.get("m_nBaseVertex", 0) + vb_base[vb]
                if len(idx) < 3:
                    continue
                material = call.get("m_material", "")
                if material not in materials:
                    materials.append(material)
                submeshes.append(idx[:len(idx) // 3 * 3].reshape(-1, 3))
                slots.append(materials.index(material))
    if not submeshes:
        raise ValueError("This model has no triangles (it may only hold physics or animations).")
    pts = z_up_to_y_up(np.concatenate(points))
    nrm = z_up_to_y_up(np.concatenate(normals))
    uv = np.concatenate(uvs).astype(np.float32)
    uv[:, 1] = 1.0 - uv[:, 1]
    has_normals = bool(np.any(nrm))
    subs = [orient_to_normals(pts, s, nrm) if has_normals else s for s in submeshes]
    md = MeshData(pts, subs, normals=nrm if has_normals else None, uvs={"UV0": uv}, material_slots=slots, name=name)
    return md, materials


# --------------------------------------------------------------------------- sounds (.vsnd_c)

def _wav(pcm, rate, channels, bits, fmt=1, fmt_chunk=None):
    if fmt_chunk is None:
        fmt_chunk = struct.pack("<HHIIHH", fmt, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def sound_audio(data):
    """.vsnd_c -> (playable bytes, extension, info rows). Follows ValveResourceFormat's Sound.cs."""
    blocks = resource_block_list(data)
    version = struct.unpack_from("<H", data, 6)[0]
    ctrl = block_kv3(data, blocks, "CTRL") if any(k == "CTRL" for k, _o, _s in blocks) else None
    header = b""
    if ctrl and ctrl.get("_class") in ("CVoiceContainerDefault", "CVoiceContainerEnvelope"):
        snd = ctrl.get("m_vSound") or {}
        fmt = {"MP3": "mp3", "PCM8": "pcm8", "PCM16": "pcm16"}.get(str(snd.get("m_nFormat")), "")
        rate, channels = snd.get("m_nRate", 44100), snd.get("m_nChannels", 1)
        size = snd.get("m_nStreamingSize", 0)
        duration = snd.get("m_flDuration", 0.0)
        start = struct.unpack_from("<I", data, 0)[0]  # sound data follows the resource
    else:
        off, bsize = next(((o, s) for k, o, s in blocks if k == "DATA"), (None, 0))
        if off is None:
            raise ValueError("Not a sound resource")
        p = off
        if version >= 4:
            rate, kind, channels = struct.unpack_from("<HBB", data, p)
            p += 4
            fmt = {0: "pcm16", 1: "pcm8", 2: "mp3", 3: "adpcm"}.get(kind, "")
        else:
            bits_info, = struct.unpack_from("<I", data, p)
            p += 4
            kind = bits_info & 3
            bits = (bits_info >> 2) & 31
            channels = (bits_info >> 7) & 3
            wave_fmt = (bits_info >> 12) & 3
            rate = (bits_info >> 14) & 0x1FFFF
            fmt = "mp3" if kind == 2 else ("adpcm" if wave_fmt == 2 else ("pcm8" if bits == 8 else "pcm16")) if kind else "aac"
        _loop, _samples, duration = struct.unpack_from("<iIf", data, p)
        p += 12
        p += 4  # sentence offset
        header_pos = p
        header_off, header_size, size = struct.unpack_from("<IiI", data, p)
        if header_size > 0:
            header = data[header_pos + header_off:header_pos + header_off + header_size]
        start = off + bsize
    raw = data[start:start + size]
    if not raw:
        raise ValueError("This sound has no audio data in the file.")
    rows = [("Format", fmt.upper()), ("Channels", str(channels)), ("Sample rate", f"{rate} Hz"),
            ("Duration", f"{duration:.2f} s")]
    if fmt == "mp3":
        return raw, "mp3", rows
    if fmt in ("pcm16", "pcm8"):
        return _wav(raw, rate, channels, 16 if fmt == "pcm16" else 8), "wav", rows
    if fmt == "adpcm" and header:
        return _wav(raw, rate, channels, 16, fmt_chunk=header), "wav", rows
    if fmt == "aac":
        return raw, "aac", rows
    raise ValueError(f"Sound format {fmt or '?'} isn't supported")


def material_textures(data):
    """.vmat_c -> [(param name, texture path)]."""
    blocks = resource_block_list(data)
    info = block_kv3(data, blocks, "DATA") or {}
    out = []
    for p in info.get("m_textureParams") or []:
        value = p.get("m_pValue")
        if isinstance(value, str) and value:
            out.append((p.get("m_name", ""), value))
    return out


# VTexFormat
VTEX_FORMATS = {
    1: "DXT1", 2: "DXT5", 3: "I8", 4: "RGBA8888", 5: "R16", 6: "RG1616", 7: "RGBA16161616", 8: "R16F",
    9: "RG1616F", 10: "RGBA16161616F", 11: "R32F", 12: "RG3232F", 13: "RGB323232F", 14: "RGBA32323232F",
    15: "JPEG_RGBA8888", 16: "PNG_RGBA8888", 17: "JPEG_DXT5", 18: "PNG_DXT5", 19: "BC6H", 20: "BC7",
    21: "ATI2N", 22: "IA88", 23: "ETC2", 24: "ETC2_EAC", 25: "R11_EAC", 26: "RG1111_EAC", 27: "ATI1N",
    28: "BGRA8888",
}
BLOCK = {1: (1, 8), 2: (3, 16), 27: (4, 8), 21: (5, 16), 19: (6, 16), 20: (7, 16)}  # fmt: (bcn, bytes/block)
BPP = {3: 1, 4: 4, 5: 2, 6: 4, 7: 8, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 12, 14: 16, 22: 2, 28: 4}
VTEX_CUBE = 0x10
EXTRA_COMPRESSED_MIPS = 4
NO_ALPHA = 8  # an alpha channel never above this (BC7 leaves stray 1-4s in an all-0 one) isn't opacity


def vtex_info(data):
    blocks = resource_blocks(data)
    if "DATA" not in blocks:
        raise ValueError("Texture has no DATA block")
    off, size = blocks["DATA"]
    (_version, flags, _r0, _r1, _r2, _r3, width, height, depth, fmt, mips, _picmip, extra_offset,
     extra_count) = struct.unpack_from("<HH4fHHHBBIII", data, off)
    info = {"flags": flags, "w": width, "h": height, "depth": max(depth, 1), "format": fmt,
            "mips": max(mips, 1), "data_offset": off + size, "compressed_mips": None}
    # Extra data: look for per-mip compressed sizes (LZ4).
    pos = off + 32 + extra_offset  # the offset counts from its own field
    for i in range(extra_count):
        kind, rel, _size = struct.unpack_from("<III", data, pos + i * 12)
        start = pos + i * 12 + 4 + rel
        if kind == EXTRA_COMPRESSED_MIPS:
            is_compressed, mips_off, n = struct.unpack_from("<III", data, start)
            if is_compressed == 1:
                sizes_at = start + 4 + mips_off
                info["compressed_mips"] = list(struct.unpack_from(f"<{n}I", data, sizes_at))
    return info


def mip_size(fmt, w, h, d=1):
    if fmt in BLOCK:
        return ((w + 3) // 4) * ((h + 3) // 4) * BLOCK[fmt][1] * d
    if fmt in BPP:
        return w * h * d * BPP[fmt]
    raise ValueError(f"Texture format {VTEX_FORMATS.get(fmt, fmt)} isn't supported yet")


def decode(fmt, data, w, h):
    if fmt in BLOCK:
        n = BLOCK[fmt][0]
        pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
        mode = {4: "L", 5: "RGB", 6: "RGB"}.get(n, "RGBA")
        args = (n, "BC6H") if n == 6 else (n,)  # BC6H: unsigned half floats
        img = Image.frombytes(mode, (pw, ph), data, "bcn", *args)
        if n == 5:
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            arr[..., 2] = np.sqrt(np.clip(1 - arr[..., 0] ** 2 - arr[..., 1] ** 2, 0, 1))
            img = Image.fromarray(((arr + 1) * 127.5).clip(0, 255).astype(np.uint8), "RGB")
        return img.crop((0, 0, w, h)) if (pw, ph) != (w, h) else img
    if fmt == 4:
        return Image.frombytes("RGBA", (w, h), data)
    if fmt == 28:
        return Image.frombytes("RGBA", (w, h), data, "raw", "BGRA")
    if fmt == 3:
        return Image.frombytes("L", (w, h), data)
    if fmt == 22:
        return Image.frombytes("LA", (w, h), data)
    dtype, ch = {5: ("<u2", 1), 6: ("<u2", 2), 7: ("<u2", 4), 8: ("<f2", 1), 9: ("<f2", 2), 10: ("<f2", 4),
                 11: ("<f4", 1), 12: ("<f4", 2), 13: ("<f4", 3), 14: ("<f4", 4)}[fmt]
    arr = np.frombuffer(data, dtype, w * h * ch).reshape(h, w, ch).astype(np.float32)
    if dtype == "<u2":
        arr /= 65535.0
    else:
        arr = arr / (1.0 + np.abs(arr))  # tonemap HDR
    if ch < 3:
        arr = np.concatenate([arr, np.zeros((h, w, 3 - ch), np.float32)], axis=2)
    return Image.fromarray((np.clip(arr[..., :3], 0, 1) * 255).astype(np.uint8), "RGB")


def vtex_image(data):
    info = vtex_info(data)
    fmt, w, h, depth = info["format"], info["w"], info["h"], info["depth"]
    if fmt in (15, 16, 17, 18):  # stored as a whole JPEG/PNG file
        return pil_image_from_bytes(data[info["data_offset"]:])
    faces = 6 if info["flags"] & VTEX_CUBE else 1
    pos = info["data_offset"]
    compressed = info["compressed_mips"]
    # Mips are stored smallest first; mip 0 is last.
    for mip in range(info["mips"] - 1, 0, -1):
        if compressed:
            pos += compressed[mip]
        else:
            pos += mip_size(fmt, max(1, w >> mip), max(1, h >> mip), max(1, depth >> mip)) * faces
    size = mip_size(fmt, w, h, depth) * faces
    if compressed and compressed[0] < size:  # a mip that didn't shrink is stored as-is
        import lz4.block
        raw = lz4.block.decompress(data[pos:pos + compressed[0]], uncompressed_size=size)
    else:
        raw = data[pos:pos + size]
    if len(raw) < mip_size(fmt, w, h):
        raise ValueError("Texture data is cut short")
    return decode(fmt, raw[:mip_size(fmt, w, h)], w, h)


# --------------------------------------------------------------------------- session

class Source2Session(GameSession):
    def __init__(self, plugin, path, fs):
        super().__init__(plugin, path)
        self.fs = fs
        self.by_key = {}

    def close(self):
        self.fs.close()

    def raw(self, asset):
        if asset.kind == "data":
            content, _ext, _refs = resource_data(self.fs.read(asset.ref))
            if isinstance(content, str):
                return content.encode("utf-8")
            import json
            return json.dumps(_json_safe(content), indent=2, ensure_ascii=False).encode("utf-8")
        return self.fs.read(asset.ref)

    def text(self, asset):
        if asset.kind != "data":
            return super().text(asset)
        import json
        content, _ext, refs = resource_data(self.fs.read(asset.ref))
        ext = asset.ref.rsplit(".", 1)[-1]
        head = [f"{asset.name}   ({DATA_TYPES.get(ext, ext)})"]
        tail = []
        if refs:  # after the data: the data is what people look for
            tail = ["", "", f"Uses {len(refs)} file(s):"] + [f"  {r}" for r in refs[:2000]]
            if len(refs) > 2000:
                tail.append(f"  ... and {len(refs) - 2000} more")
        if isinstance(content, str):
            body = content
        elif isinstance(content, list) and content and all(isinstance(c, str) for c in content):
            body = "Text found in this file (its format isn't decoded yet):\n\n" + "\n".join(content)
        else:
            body = json.dumps(_json_safe(content), indent=2, ensure_ascii=False)
        return "\n".join(head) + "\n\n" + body + "\n".join(tail)

    def image(self, asset):
        data = self.fs.read(asset.ref)
        if asset.ref.endswith(".vtex_c"):
            img = vtex_image(data)
            if img.mode == "RGBA" and img.getchannel("A").getextrema()[1] <= NO_ALPHA:
                img = img.convert("RGB")  # alpha ~0 everywhere: the channel holds no opacity (Deadlock g_tColor)
            return img
        if asset.ref.endswith(".vsvg_c"):
            from engines.sdk import svg_image
            text = panorama_text(data, resource_block_list(data))
            if not text:
                raise ValueError("This SVG has no image data.")
            return svg_image(text.encode("utf-8"))
        return pil_image_from_bytes(data)

    def audio(self, asset):
        if asset.ref.endswith(".vsnd_c"):
            data, ext, _rows = sound_audio(self.fs.read(asset.ref))
            return data, ext
        return super().audio(asset)

    def mesh(self, asset):
        md, _materials = model_meshdata(self.fs.read(asset.ref), name=asset.name.rsplit("/", 1)[-1])
        return md

    ALBEDO_PARAMS = ("g_tcolor", "g_tcolor1", "g_tcolora", "g_tbasecolor", "g_talbedo", "g_tlayer1color")
    NORMAL_PARAMS = ("g_tnormal", "g_tnormal1", "g_tnormala", "g_tnormalroughness", "g_tlayer1normal")

    def materials(self, asset):
        from engines.sdk import ALBEDO, NORMAL, OTHER, Material, TextureRef
        _md, paths = model_meshdata(self.fs.read(asset.ref))
        out = []
        for path in paths:
            key = (path.lower() + "_c") if path else ""
            refs = []
            if key and self.fs.exists(key):
                try:
                    params = material_textures(self.fs.read(key))
                except Exception as e:
                    log.debug("Couldn't read material %s: %s", path, e)
                    params = []
                for param, tex in params:
                    tex_key = tex.lower() + ("" if tex.lower().endswith("_c") else "_c")
                    tex_asset = self.by_key.get(tex_key)
                    if tex_asset is None:
                        continue
                    low = param.lower()
                    role = ALBEDO if low in self.ALBEDO_PARAMS else NORMAL if low in self.NORMAL_PARAMS else OTHER
                    refs.append(TextureRef(param, tex_asset.name.rsplit("/", 1)[-1], tex_asset, role))
                refs.sort(key=lambda r: (r.role != ALBEDO, r.role == NORMAL))
            out.append(Material(path.rsplit("/", 1)[-1] if path else "(no material)", refs))
        return out

    def stats(self, asset):
        stats = {"size": asset.size, "info": "", "sort": asset.size or 0}
        if asset.kind == "data":
            ext = asset.ref.rsplit(".", 1)[-1]
            stats["info"] = DATA_TYPES.get(ext, ext)
            return stats
        if asset.kind == "model":
            md = self.mesh(asset)
            tris = md.triangle_count
            stats.update(tris=tris, verts=len(md.points), info=f"{tris:,} tris", sort=tris)
            return stats
        if asset.ref.endswith(".vtex_c"):
            info = vtex_info(self.fs.read(asset.ref, 0, 4096))
            w, h = info["w"], info["h"]
            stats.update(w=w, h=h, info=f"{w}×{h}", sort=w * h)
        return stats

    def describe(self, asset):
        rows = [("File", self.fs.source_name(asset.ref)), ("Path", asset.path)]
        if asset.ref.endswith(".vsnd_c"):
            try:
                rows += sound_audio(self.fs.read(asset.ref))[2]
            except Exception:
                pass
        if asset.ref.endswith(".vtex_c"):
            try:
                info = vtex_info(self.fs.read(asset.ref, 0, 4096))
                rows.append(("Format", VTEX_FORMATS.get(info["format"], str(info["format"]))))
            except Exception:
                pass
        return rows


class Source2Plugin(EnginePlugin):
    id = "source2"
    name = "Source 2"
    version = "1.0"
    author = "UniView"
    description = ("Source 2 games (CS2, Dota 2, HL: Alyx, Deadlock...): VPK packs, .vmdl_c models with "
                   "their materials, .vtex_c textures.")

    def detect(self, path):
        if os.path.isfile(path):
            if path.lower().endswith("_dir.vpk") and os.path.isfile(os.path.join(os.path.dirname(path), "gameinfo.gi")):
                return 96
            return 0
        mods = mod_dirs(path)
        return 96 if any("gameinfo.gi" in listdir_lower(m) for m in mods) else 0

    def game_info(self, path):
        mods = mod_dirs(path)
        version = ""
        for mod in mods:
            try:
                with open(os.path.join(mod, "steam.inf"), encoding="utf-8", errors="replace") as f:
                    match = re.search(r"(?:ClientVersion|PatchVersion)\s*=\s*([\d.]+)", f.read())
                if match:
                    version = match.group(1)
                    break
            except OSError:
                continue
        return {"engine_version": "", "detail": ", ".join(os.path.basename(m) for m in mods)
                + (f" · build {version}" if version else "")}

    def count_files(self, path):
        if os.path.isfile(path):
            return 1
        return sum(len(dir_vpks(m)) for m in mod_dirs(path))

    def open(self, path, progress):
        fs = VirtualFS()
        vpk_files = [path] if os.path.isfile(path) else [v for m in mod_dirs(path) for v in dir_vpks(m)]
        if not vpk_files:
            raise FileNotFoundError(f"No Source 2 VPK packs found in:\n{path}")
        failed = 0
        progress.files((os.path.basename(v), vpk_size(v), v) for v in vpk_files)
        for n, vpk_path in enumerate(vpk_files, 1):
            progress.file(n - 1)
            progress(f"Reading {os.path.basename(os.path.dirname(vpk_path))}/{os.path.basename(vpk_path)} "
                     f"({n}/{len(vpk_files)})", n - 1, len(vpk_files))
            try:
                fs.add_vpk(VPK(vpk_path))
            except Exception as e:
                failed += 1
                log.warning("Could not read %s: %s", vpk_path, e)
        progress.file(len(vpk_files))
        progress("Listing assets ...")
        session = Source2Session(self, path, fs)
        session.file_count = len(fs.packs)
        for key in fs.files:
            display = fs.display_path(key)
            base = key.rsplit("/", 1)[-1]
            ext = base.rsplit(".", 1)[-1] if "." in base else ""
            kind: Kind
            if ext == "vtex_c":
                kind, name = "texture", display[:-len(".vtex_c")]
                if name.lower().startswith("materials/"):
                    name = name[len("materials/"):]
            elif ext == "vmdl_c":
                kind, name = "model", display[:-len(".vmdl_c")]
                if name.lower().startswith("models/"):
                    name = name[len("models/"):]
            elif ext == "vsvg_c":
                kind, name = "texture", display[:-len("_c")]
            elif ext.endswith("_c") and ext not in ("vsnd_c",):
                # Compiled resources: shown as their decoded data; saved as JSON / the UI file's text.
                kind, name = "data", display[:-len("_c")]
                ext = PANORAMA_EXTS.get(ext) or "json"
            else:
                kind, name = kind_for_extension(ext), display
            asset = Asset(kind, name, key, uid=key, size=fs.size(key), path=display, ref=key, ext=ext or "bin")
            session.assets.append(asset)
            session.by_key[key] = asset
        if failed:
            session.warnings.append(f"{failed} VPK file(s) could not be read (see the console).")
        return session


PLUGIN = Source2Plugin()
