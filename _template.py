"""Template engine plugin for UniView - copy this file to start your own.

    1. Copy it to plugins/my_engine.py (files starting with "_" are ignored, so this one never loads).
    2. Change the id/name, then fill in detect(), open() and the GameSession methods you need.
    3. Restart UniView and check Options -> Plugin manager (load errors show there and in the console).
       To give your plugin to someone else, send them the .py (or a .zip with it and its helper files): they
       install it with Options -> Plugin manager -> Install plugin..., or by dropping it on that window.

Every .py directly in plugins/ is loaded as a plugin and must set PLUGIN = ... (or PLUGINS = [...]) at the top
level. Keep helper modules in a subfolder (a package next to your plugin) or start their names with "_".

As written it's a working "loose files" engine: it shows .png/.jpg/.tga/.dds images, text files and
Wavefront .obj models found anywhere under a folder. That makes it a handy starting point for a game
that keeps some of its files unpacked. Read engines/sdk.py (in UniView) for the full API and source.py,
unreal.py or spore.py in the UniView-Plugins repo (https://github.com/NightHawkHSI/UniView-Plugins) for real
examples (archives, compression, binary formats, materials). Shared readers other plugins can use live there too:
valve_formats/ (vpk, bsp, kv3, meshopt) and unreal_formats/ (ue_pak, iostore, ue_package, aes).

Plugins published in UniView-Plugins only read game files and write nothing but caches (engines.sdk.cache_dir());
no network, no subprocess, no eval/exec. Users run plugins as local code, so keep yours the same.

Only the standard library, numpy and Pillow (PIL) are guaranteed to be available, also in the
packaged UniView.exe. Anything else you import has to be installed next to it yourself.

The type hints are optional, but with them an editor (or `py -m mypy plugins/my_engine.py`) tells
you when a method returns the wrong shape - see the TypedDicts at the top of engines/sdk.py.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from engines.sdk import (
    ALBEDO, Asset, AssetStats, EnginePlugin, GameInfo, GameSession, Kind, Material, MeshData, Progress,
    TextureRef, kind_for_extension, pil_image_from_bytes,
)

if TYPE_CHECKING:
    from PIL.Image import Image


class TemplateSession(GameSession):
    """One loaded game. self.assets is filled by the plugin's open()."""

    def raw(self, asset: Asset) -> bytes:
        # asset.ref is whatever you stored when listing the asset - here, the real file path.
        with open(asset.ref, "rb") as f:
            return f.read()

    def image(self, asset: Asset) -> Image:
        # Return a PIL image. Decode your engine's texture format here.
        return pil_image_from_bytes(self.raw(asset))

    def mesh(self, asset: Asset) -> MeshData:
        # Return MeshData: right-handed, Y up, counter-clockwise triangles, UV origin bottom-left.
        # (Use engines.sdk.z_up_to_y_up() for Z-up engines, and flip v for DirectX-style UVs.)
        points: list[list[float]] = []
        uvs: list[list[float]] = []
        tris: list[list[int]] = []
        face_uvs: dict[int, int] = {}
        with open(asset.ref, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    points.append([float(x) for x in parts[1:4]])
                elif parts[0] == "vt":
                    uvs.append([float(x) for x in parts[1:3]])
                elif parts[0] == "f":
                    corners = [p.split("/") for p in parts[1:]]
                    ids = [int(c[0]) - 1 if int(c[0]) > 0 else len(points) + int(c[0]) for c in corners]
                    for c, vid in zip(corners, ids):
                        if len(c) > 1 and c[1]:
                            face_uvs[vid] = int(c[1]) - 1
                    for i in range(1, len(ids) - 1):  # fan-triangulate polygons
                        tris.append([ids[0], ids[i], ids[i + 1]])
        uv0 = None
        if uvs and face_uvs:
            uv0 = np.zeros((len(points), 2), np.float32)
            for vid, tid in face_uvs.items():
                if tid < len(uvs):
                    uv0[vid] = uvs[tid]
        return MeshData(np.array(points, np.float32), [np.array(tris)],
                        uvs={"UV0": uv0} if uv0 is not None else None,
                        name=os.path.splitext(os.path.basename(asset.ref))[0])

    def materials(self, asset: Asset) -> list[Material]:
        # Optional: [Material(name, [TextureRef(slot, name, texture Asset, role)])].
        # The first ALBEDO texture is shown on the model and exported with it.
        # Here: a texture next to the .obj with the same name, if there is one.
        stem = os.path.splitext(asset.ref)[0].lower()
        for tex in self.assets:
            if tex.kind == "texture" and os.path.splitext(tex.ref)[0].lower() == stem:
                return [Material(os.path.basename(stem), [TextureRef("diffuse", tex.name, tex, ALBEDO)])]
        return []

    def stats(self, asset: Asset) -> AssetStats:
        # Optional: cheap numbers for sorting and search filters (tris>1000, w>=512 ...).
        return {"size": asset.size, "info": "", "sort": asset.size or 0}


class TemplatePlugin(EnginePlugin):
    id = "loose_files"          # unique; saved with each game. Same id as a built-in = replaces it.
    name = "Loose files"
    version = "1.0"
    author = "you"
    description = "Images, text files and .obj models lying loose in a folder."

    def detect(self, path: str) -> int:
        # 0 = not mine ... 100 = certainly mine. Keep it fast (folder listings, a few file headers).
        # Real engine plugins return 90+ when they see their marker files; this one stays low so it
        # only wins when nothing else recognizes the folder.
        return 5 if os.path.isdir(path) else 0

    def game_info(self, path: str) -> GameInfo:
        # Shown on the game's box on the Projects page. Only read small things here.
        return {"engine_version": "", "detail": "loose files"}

    def count_files(self, path: str) -> int:
        return sum(len(files) for _root, _dirs, files in os.walk(path))

    def open(self, path: str, progress: Progress) -> TemplateSession:
        # Runs on a worker thread. progress(text, done, total) updates the loading screen.
        session = TemplateSession(self, path)
        progress("Listing files ...")
        for root, _dirs, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, path).replace("\\", "/")
                ext = os.path.splitext(name)[1].lower().lstrip(".")
                kind: Kind = "model" if ext == "obj" else kind_for_extension(ext)
                if kind == "file":
                    continue  # skip what we can't show
                session.assets.append(Asset(
                    kind=kind,                                   # model/texture/sprite/text/audio/file
                    name=rel.rsplit(".", 1)[0] if kind in ("model", "texture") else rel,
                    key=rel.lower(),                             # unique inside this game
                    uid=rel.lower(),                             # stable between runs (favorites)
                    size=os.path.getsize(full),
                    path=rel,                                    # used for "keep folder structure" exports
                    source=os.path.basename(root),
                    ref=full,                                    # your own handle to find the data again
                    ext=ext,
                ))
        session.file_count = len(session.assets)
        return session


PLUGIN = TemplatePlugin()
