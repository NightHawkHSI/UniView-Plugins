"""Unreal IoStore containers (.utoc + .ucas), used by UE5 and later UE4 games.

The .utoc is the table of contents (chunk ids, offsets, compression blocks, file names); the
.ucas holds the compressed chunks. A chunk is one package (.uasset/.umap), one bulk-data file
(.ubulk/.uptnl), a shader library...
"""

import struct
import threading

TOC_MAGIC = b"-==--==--==--==-"
FLAG_COMPRESSED, FLAG_ENCRYPTED, FLAG_SIGNED, FLAG_INDEXED = 1, 2, 4, 8

# EIoChunkType (UE5)
CHUNK_EXPORT_BUNDLE_DATA = 1
CHUNK_BULK_DATA = 2
CHUNK_OPTIONAL_BULK_DATA = 3
CHUNK_MEMORY_MAPPED_BULK_DATA = 4
CHUNK_SCRIPT_OBJECTS = 5
CHUNK_CONTAINER_HEADER = 6


def _u40(b, i):
    """5-byte big-endian integer (FIoOffsetAndLength)."""
    return int.from_bytes(b[i:i + 5], "big")


class TocChunk:
    __slots__ = ("index", "id", "type", "offset", "length", "path")

    def __init__(self, index, chunk_id, chunk_type, offset, length):
        self.index, self.id, self.type, self.offset, self.length = index, chunk_id, chunk_type, offset, length
        self.path = None


class IoStore:
    """One .utoc/.ucas pair. read() is thread-safe."""

    def __init__(self, utoc_path, decompress, aes=None):
        """decompress(method name, data, uncompressed size) -> bytes; aes: unreal_formats.aes.AES or None."""
        self.utoc_path = utoc_path
        self.decompress = decompress
        self.aes = aes
        self._lock = threading.Lock()
        self._handles = {}
        self._cache = {}  # block index -> decompressed bytes
        with open(utoc_path, "rb") as f:
            data = f.read()
        if data[:16] != TOC_MAGIC:
            raise ValueError("Not an IoStore .utoc file")
        self.version = data[16]
        (header_size, entry_count, block_count, block_entry_size, method_count, method_len, self.block_size,
         dir_size, self.partition_count) = struct.unpack_from("<9I", data, 20)
        self.container_id, = struct.unpack_from("<Q", data, 56)
        self.encryption_guid = data[64:80]
        self.flags = data[80]
        seeds_count, = struct.unpack_from("<I", data, 84)
        self.partition_size, = struct.unpack_from("<Q", data, 88)
        no_hash_count, = struct.unpack_from("<I", data, 96)
        if self.version < 3 or not self.partition_size:
            self.partition_size = 1 << 62
        pos = header_size

        ids = data[pos:pos + entry_count * 12]
        pos += entry_count * 12
        offs = data[pos:pos + entry_count * 10]
        pos += entry_count * 10
        if self.version >= 4:
            pos += seeds_count * 4
        if self.version >= 5 and no_hash_count != 0xFFFFFFFF:
            pos += no_hash_count * 4
        self.blocks = []  # (offset in .ucas space, compressed size, uncompressed size, method index)
        for i in range(block_count):
            b = data[pos + i * block_entry_size: pos + i * block_entry_size + 12]
            offset = int.from_bytes(b[0:5], "little")
            csize = int.from_bytes(b[5:8], "little")
            usize = int.from_bytes(b[8:11], "little")
            self.blocks.append((offset, csize, usize, b[11]))
        pos += block_count * block_entry_size
        self.methods = [""]
        for i in range(method_count):
            name = data[pos + i * method_len: pos + (i + 1) * method_len].split(b"\0")[0]
            self.methods.append(name.decode("ascii", "replace").lower())
        pos += method_count * method_len
        if self.flags & FLAG_SIGNED:
            hash_size, = struct.unpack_from("<i", data, pos)
            pos += 4 + hash_size * 2 + block_count * 20
        self.chunks = []
        for i in range(entry_count):
            chunk_id = ids[i * 12:i * 12 + 8]
            chunk_type = ids[i * 12 + 11]
            self.chunks.append(TocChunk(i, chunk_id, chunk_type, _u40(offs, i * 10), _u40(offs, i * 10 + 5)))
        self.files = {}  # path -> TocChunk
        if self.flags & FLAG_INDEXED and dir_size:
            index = data[pos:pos + dir_size]
            if self.flags & FLAG_ENCRYPTED:
                if self.aes is None:
                    raise PermissionError("the container is encrypted (needs the game's AES key)")
                index = self.aes.decrypt(index)
            self._read_directory(index)

    @property
    def encrypted(self):
        return bool(self.flags & FLAG_ENCRYPTED)

    def _read_directory(self, data):
        pos = 0

        def fstring():
            nonlocal pos
            n, = struct.unpack_from("<i", data, pos)
            pos += 4
            if n == 0:
                return ""
            if n < 0:
                raw = data[pos:pos - n * 2]
                pos += -n * 2
                return raw.decode("utf-16-le", "replace").rstrip("\0")
            raw = data[pos:pos + n]
            pos += n
            return raw.decode("utf-8", "replace").rstrip("\0")

        mount = fstring().replace("\\", "/")
        while mount.startswith("../"):
            mount = mount[3:]
        mount = mount.lstrip("/")
        n_dirs, = struct.unpack_from("<i", data, pos)
        pos += 4
        dirs = [struct.unpack_from("<4I", data, pos + i * 16) for i in range(n_dirs)]
        pos += n_dirs * 16
        n_files, = struct.unpack_from("<i", data, pos)
        pos += 4
        files = [struct.unpack_from("<3I", data, pos + i * 12) for i in range(n_files)]
        pos += n_files * 12
        n_strings, = struct.unpack_from("<i", data, pos)
        pos += 4
        strings = [fstring() for _ in range(n_strings)]
        none = 0xFFFFFFFF
        stack = [(0, mount)] if dirs else []
        while stack:
            d, prefix = stack.pop()
            name, first_child, next_sibling, first_file = dirs[d]
            path = prefix + (strings[name] + "/" if name != none and name < len(strings) else "")
            f = first_file
            while f != none and f < len(files):
                fname, next_file, user_data = files[f]
                if user_data < len(self.chunks):
                    chunk = self.chunks[user_data]
                    chunk.path = path + strings[fname]
                    self.files[chunk.path.lower()] = chunk
                f = next_file
            if next_sibling != none:
                stack.append((next_sibling, prefix))
            if first_child != none:
                stack.append((first_child, path))

    def chunk_of_type(self, chunk_type):
        return next((c for c in self.chunks if c.type == chunk_type), None)

    def _partition_path(self, index):
        base = self.utoc_path[:-5]
        return f"{base}.ucas" if index == 0 else f"{base}_s{index}.ucas"

    def _read_raw(self, offset, size):
        part, offset = divmod(offset, self.partition_size)
        with self._lock:
            f = self._handles.get(part)
            if f is None:
                f = self._handles[part] = open(self._partition_path(part), "rb")
            f.seek(offset)
            return f.read(size)

    def read(self, chunk, start=0, length=None):
        """Uncompressed bytes of a chunk (or a slice of it)."""
        total = chunk.length
        if length is None or start + length > total:
            length = max(0, total - start)
        if not length:
            return b""
        begin = chunk.offset + start
        end = begin + length
        first, last = begin // self.block_size, (end - 1) // self.block_size
        out = bytearray()
        for b in range(first, last + 1):
            out += self._block(b)
        skip = begin - first * self.block_size
        return bytes(out[skip:skip + length])

    CACHE_BLOCKS = 32

    def _block(self, b):
        """One decompressed block (a few recent ones are cached: small packages share blocks)."""
        cached = self._cache.get(b)
        if cached is not None:
            return cached
        offset, csize, usize, method = self.blocks[b]
        raw = self._read_raw(offset, (csize + 15) & ~15 if self.encrypted else csize)
        if self.encrypted:
            if self.aes is None:
                raise PermissionError("This container is encrypted (needs the game's AES key).")
            raw = self.aes.decrypt(raw)
        raw = raw[:csize]
        if method:
            raw = self.decompress(self.methods[method], raw, usize)
        raw = bytes(raw[:usize])
        with self._lock:
            self._cache[b] = raw
            while len(self._cache) > self.CACHE_BLOCKS:
                self._cache.pop(next(iter(self._cache)))
        return raw

    def close(self):
        with self._lock:
            for f in self._handles.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._handles = {}
