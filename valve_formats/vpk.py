"""Valve VPK archives (v1 and v2), used by Source and Source 2 games.

A pack is <name>_dir.vpk (the directory tree, sometimes with data) plus <name>_000.vpk, _001 ...
"""

import os
import struct
import threading

VPK_MAGIC = 0x55AA1234
DIR_ARCHIVE = 0x7FFF


def vpk_size(dir_path):
    """Bytes in a pack: the _dir.vpk plus its numbered archives (name_000.vpk ...)."""
    import glob
    total = 0
    stem = dir_path[:-len("_dir.vpk")] if dir_path.lower().endswith("_dir.vpk") else None
    for p in [dir_path] + (glob.glob(glob.escape(stem) + "_[0-9][0-9][0-9].vpk") if stem else []):
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


class VPKEntry:
    __slots__ = ("pack", "path", "crc", "preload", "archive", "offset", "length")

    def __init__(self, pack, path, crc, preload, archive, offset, length):
        self.pack, self.path, self.crc = pack, path, crc
        self.preload, self.archive, self.offset, self.length = preload, archive, offset, length

    @property
    def size(self):
        return len(self.preload) + self.length


class VPK:
    """One _dir.vpk and its numbered archives. read() is thread-safe."""

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.prefix = dir_path[: -len("_dir.vpk")]
        self.entries = {}   # lowercase 'dir/name.ext' -> VPKEntry
        self._handles = {}
        self._lock = threading.Lock()
        with open(dir_path, "rb") as f:
            data = f.read()
        magic, self.version, tree_size = struct.unpack_from("<III", data, 0)
        if magic != VPK_MAGIC:
            raise ValueError(f"{os.path.basename(dir_path)} is not a VPK file")
        header = 12 if self.version == 1 else 28
        if self.version not in (1, 2):
            raise ValueError(f"Unsupported VPK version {self.version}")
        self.data_start = header + tree_size
        self._parse_tree(data, header, header + tree_size)

    def _parse_tree(self, data, pos, end):
        def cstr():
            nonlocal pos
            stop = data.index(b"\0", pos)
            text = data[pos:stop].decode("utf-8", "replace")
            pos = stop + 1
            return text

        while pos < end:
            ext = cstr()
            if not ext:
                break
            while True:
                folder = cstr()
                if not folder:
                    break
                folder = "" if folder == " " else folder.strip("/") + "/"
                while True:
                    name = cstr()
                    if not name:
                        break
                    crc, preload_len, archive, offset, length, _term = struct.unpack_from("<IHHIIH", data, pos)
                    pos += 18
                    preload = data[pos:pos + preload_len]
                    pos += preload_len
                    full = f"{folder}{name}" + (f".{ext}" if ext != " " else "")
                    self.entries[full.lower()] = VPKEntry(self, full, crc, preload, archive, offset, length)

    def _archive_path(self, index):
        if index == DIR_ARCHIVE:
            return self.dir_path
        return f"{self.prefix}_{index:03d}.vpk"

    def read(self, entry, start=0, length=None):
        """Bytes of an entry (or a slice of it: start, length)."""
        total = entry.size
        if length is None or start + length > total:
            length = max(0, total - start)
        out = bytearray()
        pre = len(entry.preload)
        if start < pre:
            out += entry.preload[start:start + length]
        if entry.length and start + length > pre:
            a_start = max(0, start - pre)
            a_len = length - len(out)
            offset = entry.offset + a_start + (self.data_start if entry.archive == DIR_ARCHIVE else 0)
            with self._lock:
                f = self._handles.get(entry.archive)
                if f is None:
                    f = self._handles[entry.archive] = open(self._archive_path(entry.archive), "rb")
                f.seek(offset)
                out += f.read(a_len)
        return bytes(out)

    def close(self):
        with self._lock:
            for f in self._handles.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._handles = {}


class VirtualFS:
    """Several VPKs plus loose folders merged into one tree; the first one added wins on duplicates."""

    def __init__(self):
        self.files = {}   # lowercase path -> (source, entry-or-real-path)
        self.packs = []

    def add_vpk(self, vpk):
        self.packs.append(vpk)
        for key, entry in vpk.entries.items():
            self.files.setdefault(key, (vpk, entry))

    def add_folder(self, root, subdirs=None, skip_exts=(".vpk",)):
        """Loose files under root (only these subfolders if given)."""
        bases = [os.path.join(root, d) for d in subdirs] if subdirs else [root]
        for base in bases:
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, names in os.walk(base):
                for name in names:
                    if name.lower().endswith(skip_exts):
                        continue
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, root).replace("\\", "/")
                    self.files.setdefault(rel.lower(), (None, full))

    def exists(self, path):
        return path.lower().replace("\\", "/") in self.files

    def display_path(self, key):
        src, entry = self.files[key]
        return entry.path if src is not None else key

    def size(self, key):
        src, entry = self.files[key]
        if src is not None:
            return entry.size
        try:
            return os.path.getsize(entry)
        except OSError:
            return None

    def source_name(self, key):
        src, entry = self.files[key]
        return os.path.basename(src.dir_path) if src is not None else os.path.basename(os.path.dirname(entry)) + " (loose file)"

    def read(self, path, start=0, length=None):
        key = path.lower().replace("\\", "/")
        src, entry = self.files[key]
        if src is not None:
            return src.read(entry, start, length)
        with open(entry, "rb") as f:
            f.seek(start)
            return f.read() if length is None else f.read(length)

    def close(self):
        for p in self.packs:
            p.close()
