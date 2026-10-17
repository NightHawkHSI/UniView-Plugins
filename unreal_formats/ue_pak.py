"""Unreal Engine archives: .pak files (versions 3-12), Oodle/zlib/gzip/LZ4 decompression, AES keys.

Used by the Unreal plugin (plugins/unreal.py) and available to any plugin: Pak(path, keys) lists and
reads the files of a .pak, find_paks()/find_utocs() find a game's archives (IoStore .utoc/.ucas
containers are read by unreal_formats.iostore). Oodle needs an oo2core_*_win64.dll: find_oodle() looks in
the game folder, next to UniView and in the Steam libraries.
"""

import ctypes
import glob
import logging
import os
import re
import struct
import sys
import threading
import zlib

from .aes import AES, parse_key

log = logging.getLogger("viewer.unreal")

PAK_MAGIC = 0x5A6F12E1
PAK_VERSIONS = {1: "UE4 (early)", 2: "UE4 (early)", 3: "UE 4.0-4.2", 4: "UE 4.3-4.15", 5: "UE 4.16-4.19",
                6: "UE 4.20", 7: "UE 4.21", 8: "UE 4.22-4.24", 9: "UE 4.25", 10: "UE 4.25",
                11: "UE 4.26-5.2", 12: "UE 5.3+"}
LEGACY_COMPRESSION = {0: "", 1: "zlib", 2: "gzip", 4: "oodle"}


# --------------------------------------------------------------------------- Oodle (optional)

_oodle = None
_oodle_lock = threading.Lock()


def steam_commons():
    steam = r"C:\Program Files (x86)\Steam"
    libs = [steam]
    try:
        with open(os.path.join(steam, "steamapps", "libraryfolders.vdf"), encoding="utf-8", errors="replace") as f:
            libs += [p.replace("\\\\", "\\") for p in re.findall(r'"path"\s+"([^"]+)"', f.read())]
    except OSError:
        pass
    return [os.path.join(lib, "steamapps", "common") for lib in dict.fromkeys(libs)
            if os.path.isdir(os.path.join(lib, "steamapps", "common"))]


def find_oodle(hint_dirs=()):
    """Path of an oo2core_*.dll: game folder, UniView folder/plugins, then Steam libraries."""
    patterns = []
    for base in hint_dirs:
        patterns += [os.path.join(base, "oo2core_*_win64.dll"), os.path.join(base, "*", "oo2core_*_win64.dll"),
                     os.path.join(base, "*", "*", "Binaries", "Win64", "oo2core_*_win64.dll"),
                     os.path.join(base, "Engine", "Binaries", "ThirdParty", "Oodle", "*", "oo2core_*_win64.dll")]
    app = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns += [os.path.join(app, "oo2core_*_win64.dll"), os.path.join(app, "plugins", "oo2core_*_win64.dll")]
    for pattern in patterns:
        found = sorted(glob.glob(pattern), reverse=True)
        if found:
            return found[0]
    for common in steam_commons():
        found = sorted(glob.glob(os.path.join(common, "*", "oo2core_*_win64.dll")), reverse=True)
        if found:
            return found[0]
    return None


def oodle(hint_dirs=()):
    """ctypes OodleLZ_Decompress, or None if no DLL can be found."""
    global _oodle
    with _oodle_lock:
        if _oodle is None:
            path = find_oodle(hint_dirs)
            _oodle = False
            if path:
                try:
                    fn = ctypes.WinDLL(path).OodleLZ_Decompress
                    fn.restype = ctypes.c_int64
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
                    _oodle = fn
                    log.info("Using Oodle from %s", path)
                except OSError as e:
                    log.warning("Could not load %s: %s", path, e)
        return _oodle or None


def oodle_decompress(data, size, hint_dirs=()):
    fn = oodle(hint_dirs)
    if fn is None:
        raise RuntimeError("This file is Oodle-compressed and no oo2core_*_win64.dll was found. "
                           "Copy one from a game that ships it next to UniView.exe.")
    src = ctypes.create_string_buffer(bytes(data), len(data))
    dst = ctypes.create_string_buffer(size)
    got = fn(src, len(data), dst, size, 1, 0, 0, None, 0, None, None, None, 0, 3)
    if got != size:
        raise RuntimeError(f"Oodle decompression failed ({got} of {size} bytes)")
    return dst.raw


def decompress(method, data, size, hint_dirs=()):
    if method == "zlib":
        return zlib.decompress(data)
    if method == "gzip":
        return zlib.decompress(data, 16 + zlib.MAX_WBITS)
    if method == "lz4":
        import lz4.block
        return lz4.block.decompress(data, uncompressed_size=size)
    if method.startswith("oodle"):
        return oodle_decompress(data, size, hint_dirs)
    raise RuntimeError(f"Compression '{method}' isn't supported")


def parse_keys(text):
    """AES keys from the user's text (one per line / comma separated, hex or base64)."""
    keys = []
    for part in re.split(r"[\s,;]+", text or ""):
        part = part.strip().strip('"')
        if not part:
            continue
        try:
            keys.append(AES(parse_key(part)))
        except ValueError as e:
            log.warning("Ignoring AES key %s...: %s", part[:10], e)
    return keys


# --------------------------------------------------------------------------- pak reading

class PakEntry:
    __slots__ = ("path", "offset", "size", "usize", "method", "encrypted", "blocks", "block_size", "header")

    def __init__(self, path, offset, size, usize, method, encrypted, blocks, block_size, header):
        self.path, self.offset, self.size, self.usize = path, offset, size, usize
        self.method, self.encrypted, self.blocks, self.block_size = method, encrypted, blocks, block_size
        self.header = header  # size of the entry record in front of the data


class Reader:
    def __init__(self, data, pos=0):
        self.data, self.pos = data, pos

    def take(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return value if len(value) > 1 else value[0]

    def bytes(self, n):
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def fstring(self):
        n = self.take("i")
        if n == 0:
            return ""
        if n < 0:
            if -n > 65536:
                raise ValueError("bad string")
            return self.bytes(-n * 2).decode("utf-16-le", "replace").rstrip("\0")
        if n > 65536:
            raise ValueError("bad string")
        return self.bytes(n).decode("utf-8", "replace").rstrip("\0")


def read_footer(f):
    f.seek(0, 2)
    size = f.tell()
    f.seek(max(0, size - 512))
    tail = f.read()
    i = tail.rfind(struct.pack("<I", PAK_MAGIC))
    if i < 0:
        raise ValueError("Not an Unreal .pak file (no footer)")
    version, index_offset, index_size = struct.unpack_from("<iqq", tail, i + 4)
    encrypted = tail[i - 1] == 1 if version >= 4 and i >= 1 else False
    names = []
    rest = tail[i + 44 + (1 if version == 9 else 0):]
    for j in range(0, len(rest) - 31, 32):
        name = rest[j:j + 32].split(b"\0")[0]
        # Some games customize the footer; only keep names that look like method names.
        names.append(name.decode("ascii").lower() if name.isascii() and name.isalnum() else "")
    return {"version": version, "index_offset": index_offset, "index_size": index_size,
            "encrypted_index": encrypted, "methods": names}


def pak_summary(path):
    with open(path, "rb") as f:
        return read_footer(f)


class Pak:
    def __init__(self, path, hint_dirs=(), keys=()):
        self.path = path
        self.hint_dirs = hint_dirs
        self.entries = []
        self.aes = None
        self._f = open(path, "rb")
        self._lock = threading.Lock()
        footer = read_footer(self._f)
        self.version = footer["version"]
        self.methods = footer["methods"]
        self.encrypted_index = footer["encrypted_index"]
        self._f.seek(footer["index_offset"])
        raw_index = self._f.read(footer["index_size"])
        candidates = [None] if not self.encrypted_index else list(keys)
        if not candidates:
            raise PermissionError("the file list is encrypted (needs the game's AES key)")
        last_error = None
        for key in candidates:
            try:
                self.aes = key
                index = key.decrypt(raw_index[:len(raw_index) // 16 * 16]) if key else raw_index
                self.entries = []
                if self.version >= 10:
                    self._read_index_v10(index)
                else:
                    self._read_index_legacy(index)
                return
            except Exception as e:
                last_error = e
        if self.encrypted_index:
            raise PermissionError("none of the AES keys opens this file (wrong key?)")
        raise last_error

    def _decrypt(self, data):
        if self.aes is None:
            raise PermissionError("This file is encrypted (needs the game's AES key).")
        return self.aes.decrypt(data)

    def _method(self, index):
        if self.version < 8:
            return LEGACY_COMPRESSION.get(index, f"method {index}")
        if index == 0:
            return ""
        return self.methods[index - 1] if index - 1 < len(self.methods) else f"method {index}"

    @staticmethod
    def _mount(mount):
        mount = mount.replace("\\", "/")
        while mount.startswith("../"):
            mount = mount[3:]
        return mount.lstrip("/")

    def _entry_size(self, method_index, n_blocks):
        size = 8 + 8 + 8 + 20 + 4
        if self.version >= 3:
            size += 1 + 4
            if method_index:
                size += 4 + 16 * n_blocks
        if self.version < 2:
            size += 8
        return size

    def _read_entry(self, r, path):
        """A full (unencoded) FPakEntry record."""
        start = r.pos
        offset, size, usize = r.take("qqq")
        method_index = r.take("I") if self.version >= 8 else r.take("i")
        if self.version < 2:
            r.take("q")  # timestamp
        r.bytes(20)
        blocks, encrypted, block_size = [], False, 0
        if self.version >= 3:
            if method_index:
                n = r.take("i")
                blocks = [r.take("qq") for _ in range(n)]
            encrypted = bool(r.take("B") & 1)
            block_size = r.take("I")
        header = r.pos - start
        if self.version >= 5:
            blocks = [(s + offset, e + offset) for s, e in blocks]
        return PakEntry(path, offset, size, usize, self._method(method_index), encrypted, blocks, block_size, header)

    def _read_index_legacy(self, data):
        r = Reader(data)
        mount = self._mount(r.fstring())
        count = r.take("i")
        if not 0 <= count <= 10_000_000:
            raise ValueError("bad pak index")
        for _ in range(count):
            name = r.fstring()
            self.entries.append(self._read_entry(r, mount + name))

    def _decode_entry(self, data, pos, path):
        r = Reader(data, pos)
        value = r.take("I")
        block_size = r.take("I") if (value & 0x3F) == 0x3F else (value & 0x3F) << 11
        method_index = (value >> 23) & 0x3F
        offset = r.take("I") if value & (1 << 31) else r.take("Q")
        usize = r.take("I") if value & (1 << 30) else r.take("Q")
        size = (r.take("I") if value & (1 << 29) else r.take("Q")) if method_index else usize
        encrypted = bool(value & (1 << 22))
        n_blocks = (value >> 6) & 0xFFFF
        header = self._entry_size(method_index, n_blocks)
        blocks = []
        if n_blocks == 1 and not encrypted:
            blocks = [(offset + header, offset + header + size)]
        elif n_blocks:
            cursor = offset + header
            for _ in range(n_blocks):
                length = r.take("I")
                blocks.append((cursor, cursor + length))
                cursor += (length + 15) // 16 * 16 if encrypted else length
        return PakEntry(path, offset, size, usize, self._method(method_index), encrypted, blocks,
                        min(block_size, usize) if block_size else usize, header)

    def _read_index_v10(self, data):
        r = Reader(data)
        mount = self._mount(r.fstring())
        r.take("i")  # entry count (the directory index below has them all)
        r.take("Q")  # path hash seed
        if r.take("I"):
            r.take("qq")
            r.bytes(20)
        has_full = r.take("I")
        full_offset = full_size = 0
        if has_full:
            full_offset, full_size = r.take("qq")
            r.bytes(20)
        encoded_size = r.take("i")
        if not 0 <= encoded_size <= len(data):
            raise ValueError("bad pak index")
        encoded = r.bytes(encoded_size)
        n_files = r.take("i")
        files = [self._read_entry(r, "") for _ in range(n_files)]
        if not has_full:
            raise ValueError("pak has no full directory index (file names were stripped)")
        self._f.seek(full_offset)
        raw = self._f.read(full_size)
        if self.aes is not None:
            raw = self.aes.decrypt(raw[:len(raw) // 16 * 16])
        d = Reader(raw)
        for _ in range(d.take("i")):
            folder = d.fstring()
            for _ in range(d.take("i")):
                name = d.fstring()
                location = d.take("i")
                path = (mount + folder.lstrip("/") + name).replace("//", "/")
                if location >= 0:
                    self.entries.append(self._decode_entry(encoded, location, path))
                elif -location - 1 < len(files):
                    entry = files[-location - 1]
                    entry.path = path
                    self.entries.append(entry)

    def read(self, entry):
        with self._lock:
            if not entry.method:
                self._f.seek(entry.offset + entry.header)
                if entry.encrypted:
                    return self._decrypt(self._f.read((entry.size + 15) // 16 * 16))[:entry.size]
                return self._f.read(entry.size)
            chunks = []
            for start, end in entry.blocks:
                self._f.seek(start)
                length = end - start
                if entry.encrypted:
                    chunks.append(self._decrypt(self._f.read((length + 15) // 16 * 16))[:length])
                else:
                    chunks.append(self._f.read(length))
        out = bytearray()
        remaining = entry.usize
        for chunk in chunks:
            want = min(entry.block_size or remaining, remaining)
            out += decompress(entry.method, chunk, want, self.hint_dirs)
            remaining -= want
        return bytes(out)

    def close(self):
        with self._lock:
            self._f.close()


# --------------------------------------------------------------------------- game layout

def find_paks(path):
    if os.path.isfile(path):
        return [path] if path.lower().endswith(".pak") else []
    found = []
    for pattern in ("*.pak", os.path.join("*", "Content", "Paks", "**", "*.pak"),
                    os.path.join("Content", "Paks", "**", "*.pak")):
        found += glob.glob(os.path.join(path, pattern), recursive=True)
    return sorted(set(found))


def find_utocs(path):
    if os.path.isfile(path):
        return [path] if path.lower().endswith(".utoc") else []
    found = []
    for pattern in ("*.utoc", os.path.join("*", "Content", "Paks", "**", "*.utoc"),
                    os.path.join("Content", "Paks", "**", "*.utoc")):
        found += glob.glob(os.path.join(path, pattern), recursive=True)
    return sorted(set(found))
