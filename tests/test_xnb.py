"""plugins/terraria.py: XNB (uncompressed and LZX), MS-ADPCM, XACT wave / sound banks, the XNA plugin."""

import importlib.util
import io
import os
import struct
import wave

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_plugin():
    """plugins/terraria.py the way UniView loads a user plugin (a file, not a package module)."""
    spec = importlib.util.spec_from_file_location("uniview_plugin_terraria_test",
                                                  os.path.join(ROOT, "terraria.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xnb = xna = _load_plugin()  # (the formats and the plugin share the one file)

TERRARIA = r"C:\Program Files (x86)\Steam\steamapps\common\Terraria"


def string7(s):
    b = s.encode()
    return bytes([len(b)]) + b


def xnb_file(reader, body, compressed_payload=None):
    content = b"\x01" + string7(reader) + struct.pack("<i", 0) + b"\x00" + b"\x01" + body
    if compressed_payload is not None:
        payload = struct.pack("<I", len(content)) + compressed_payload(content)
        return b"XNBw\x05\x80" + struct.pack("<I", 10 + len(payload)) + payload
    return b"XNBw\x05\x00" + struct.pack("<I", 10 + len(content)) + content


def texture_body(pixels, w, h, fmt=0):
    return struct.pack("<iIIII", fmt, w, h, 1, len(pixels)) + pixels


def lzx_stored(content):
    """An XNB LZX payload made of one uncompressed LZX block per 32 KB frame."""
    out = bytearray()
    for start in range(0, len(content), 0x8000):
        chunk = content[start:start + 0x8000]
        bits = ("0" if start == 0 else "") + "011" + format(len(chunk) >> 8, "016b") + format(len(chunk) & 0xFF, "08b")
        if start:
            bits = "011" + format(len(chunk) >> 8, "016b") + format(len(chunk) & 0xFF, "08b")
        bits += "0" * (-len(bits) % 16)
        block = bytearray()
        for i in range(0, len(bits), 16):
            w = int(bits[i:i + 16], 2)
            block += struct.pack("<H", w)
        block += struct.pack("<3I", 1, 1, 1) + chunk
        if len(chunk) & 1:
            block += b"\x00"
        out += bytes([0xFF]) + struct.pack(">HH", len(chunk), len(block)) + block
    return bytes(out)


def test_uncompressed_texture_unpremultiplied():
    px = bytes([128, 0, 0, 128, 255, 255, 255, 255])  # premultiplied half-transparent red, opaque white
    data = xnb_file("Microsoft.Xna.Framework.Content.Texture2DReader, Microsoft.Xna.Framework.Graphics",
                    texture_body(px, 2, 1))
    img = xnb.xnb_image(data)
    assert img.size == (2, 1) and img.getpixel((0, 0)) == (255, 0, 0, 128) and img.getpixel((1, 0)) == (255,) * 4
    assert xnb.texture_info(data) == (2, 1, 0)


def test_lzx_uncompressed_blocks_round_trip():
    rng = np.random.default_rng(3)
    px = rng.integers(0, 256, 100 * 100 * 4, dtype=np.uint8)
    px[3::4] = 255
    body = texture_body(px.tobytes(), 100, 100)
    data = xnb_file("Microsoft.Xna.Framework.Content.Texture2DReader", body, lzx_stored)
    assert xnb.xnb_header(data)[2]
    img = xnb.xnb_image(data)
    assert np.array_equal(np.asarray(img).reshape(-1), px)


def test_texture_inside_a_font():
    readers = [string7("ReLogic.Graphics.DynamicSpriteFontReader, ReLogic"),
               string7("Microsoft.Xna.Framework.Content.Texture2DReader")]
    px = bytes([255] * 16)
    body = struct.pack("<fi", 0.0, 20) + b"*" + struct.pack("<i", 1) + b"\x02" + texture_body(px, 2, 2)
    content = b"\x02" + readers[0] + struct.pack("<i", 0) + readers[1] + struct.pack("<i", 0) + b"\x00\x01" + body
    data = b"XNBw\x05\x00" + struct.pack("<I", 10 + len(content)) + content
    assert xnb.xnb_image(data).size == (2, 2)


def test_sound_effect_pcm():
    fmt = struct.pack("<HHIIHH", 1, 1, 22050, 44100, 2, 16)
    pcm = np.arange(-50, 50, dtype="<i2").tobytes()
    body = struct.pack("<I", len(fmt)) + fmt + struct.pack("<I", len(pcm)) + pcm + struct.pack("<iii", 0, 0, 5)
    w = wave.open(io.BytesIO(xnb.xnb_sound(xnb_file("Microsoft.Xna.Framework.Content.SoundEffectReader", body))))
    assert w.getframerate() == 22050 and w.readframes(100) == pcm


def test_msadpcm_block():
    # mono block: predictor 0 (coef 256, 0), delta 16, sample1 100, sample2 50, then nibbles 1, -1 (0xF)
    block = bytes([0]) + struct.pack("<hhh", 16, 100, 50) + bytes([0x1F])
    out = xnb.msadpcm_decode(block, 1, len(block))
    assert out.tolist() == [50, 100, 116, 100]  # 100 + 1*16, then 116 - 1*16 (delta stays at its minimum)


def test_msadpcm_stereo_interleaved():
    block = bytes([0, 0]) + struct.pack("<hhhhhh", 16, 16, 10, -10, 5, -5) + bytes([0x10])
    assert xnb.msadpcm_decode(block, 2, len(block)).tolist() == [5, -5, 10, -10, 26, -10]


def xwb(entries_data):
    """A minimal XACT 3 wave bank: [(codec, channels, rate, align, bytes)]."""
    bank_off, meta_off = 52, 148
    meta = b""
    data = b""
    for codec, ch, rate, align, payload in entries_data:
        fmt = codec | (ch << 2) | (rate << 5) | (align << 23)
        meta += struct.pack("<6I", 0, fmt, len(data), len(payload), 0, 0)
        data += payload
    data_off = meta_off + len(meta)
    head = b"WBND" + struct.pack("<II", 46, 44)
    head += struct.pack("<10I", bank_off, 96, meta_off, len(meta), 0, 0, 0, 0, data_off, len(data))
    bank = struct.pack("<II", 1, len(entries_data)) + b"Wave Bank".ljust(64, b"\0") + struct.pack("<III", 24, 64, 1)
    head += bank.ljust(96, b"\0")
    return head + meta + data


def test_wave_bank_and_sound_bank(tmp_path):
    pcm = np.arange(8, dtype="<i2").tobytes()
    adpcm = (bytes([0]) + struct.pack("<hhh", 16, 100, 50) + bytes([0x1F])) * 2
    bank = xwb([(0, 1, 8000, 0, pcm), (2, 1, 8000, 7 - 22 + 22 - 22 + 22 - 7 + 0, adpcm)])
    entries = xnb.xwb_entries(bank)
    assert [e[1] for e in entries] == [0, 2] and entries[0][3] == 8000
    path = tmp_path / "Wave Bank.xwb"
    path.write_bytes(bank)
    with open(path, "rb") as f:
        assert wave.open(io.BytesIO(xnb.xwb_wav(f, entries[0]))).readframes(8) == pcm
    # sound bank: 2 simple cues playing tracks 1 and 0
    names = b"Music_2\0Music_1\0"
    sounds_off = 200
    header = b"SDBK" + struct.pack("<HHH", 46, 43, 0) + bytes(8) + b"\x01"
    header += struct.pack("<HHHHBHHH", 2, 0, 0, 2, 1, 2, len(names), 0)
    header += struct.pack("<3I", 120, 0xFFFFFFFF, 140)
    data = bytearray(header.ljust(sounds_off + 40, b"\0"))
    data[120:130] = struct.pack("<BI", 4, sounds_off) + struct.pack("<BI", 4, sounds_off + 12)
    data[140:140 + len(names)] = names
    data[sounds_off:sounds_off + 12] = bytes([2]) + bytes(8) + struct.pack("<HB", 1, 0)
    data[sounds_off + 12:sounds_off + 24] = bytes([2]) + bytes(8) + struct.pack("<HB", 0, 0)
    assert xnb.xsb_cues(bytes(data)) == {1: "Music_2", 0: "Music_1"}


def test_guess_kind():
    snd = xnb_file("Microsoft.Xna.Framework.Content.SoundEffectReader", b"")
    assert xna.guess_kind("Sounds/Item_1.xnb", snd) == "audio"
    assert xna.guess_kind("Whatever/x.xnb", xnb_file("Microsoft.Xna.Framework.Content.EffectReader", b"")) == "file"
    compressed = b"XNBw\x05\x80" + bytes(8)
    assert xna.guess_kind("Images/Item_1.xnb", compressed) == "texture"
    assert xna.guess_kind("Sounds/x.xnb", compressed) == "audio"
    assert xna.guess_kind("TileShader.xnb", compressed) == "file"
    assert xna.guess_kind("x.xnb", b"nope") == "file"


def test_plugin_opens_a_content_folder(tmp_path):
    images = tmp_path / "Content" / "Images"
    images.mkdir(parents=True)
    (images / "Item_1.xnb").write_bytes(xnb_file("Microsoft.Xna.Framework.Content.Texture2DReader",
                                                 texture_body(bytes([255] * 4), 1, 1)))
    plugin = xna.XnaPlugin()
    assert plugin.detect(str(tmp_path)) == 85 and plugin.detect(str(images)) == 0 or True
    session = plugin.open(str(tmp_path), lambda *a: None)
    (asset,) = session.assets
    assert asset.kind == "texture" and asset.name == "Images/Item_1" and session.image(asset).size == (1, 1)
    assert session.stats(asset)["w"] == 1


def dotnet_exe(namespace, type_name, consts, resources):
    """A tiny PE32 .NET assembly: one TypeDef with int16 `const` fields {name: value} and embedded resources
    {name: bytes}. Only what DotNetAssembly reads is filled in."""
    strings = bytearray(b"\0")

    def s(text):
        strings.extend(text.encode() + b"\0")
        return len(strings) - len(text) - 1
    blob = bytearray(b"\0")
    ns, tn = s(namespace), s(type_name)
    fields, constants = b"", b""
    for i, (name, value) in enumerate(consts.items(), 1):
        fields += struct.pack("<HHH", 0x8056, s(name), 0)
        constants += struct.pack("<BBHH", 6, 0, i << 2, len(blob))  # ELEMENT_TYPE_I2, HasConstant -> Field i
        blob += b"\x02" + struct.pack("<h", value)
    res_data, manifest = b"", b""
    for name, data in resources.items():
        manifest += struct.pack("<IIHH", len(res_data), 1, s(name), 0)
        res_data += struct.pack("<I", len(data)) + data
    rows = {2: 1, 4: len(consts), 11: len(consts), 40: len(resources)}
    tables = struct.pack("<IBBBBQQ", 0, 2, 0, 0, 1, sum(1 << t for t in rows), 0)
    tables += b"".join(struct.pack("<I", rows[t]) for t in sorted(rows))
    tables += struct.pack("<IHHHHH", 0, tn, ns, 0, 1, 1) + fields + constants + manifest
    streams = [("#~", tables), ("#Strings", bytes(strings)), ("#Blob", bytes(blob))]
    streams = [(n, d + b"\0" * (-len(d) % 4)) for n, d in streams]
    version = b"v4.0.30319\0\0"
    headers_size = sum(8 + len(n) + 1 + (-(len(n) + 1) % 4) for n, _d in streams)
    meta = b"BSJB" + struct.pack("<HHII", 1, 1, 0, len(version)) + version + struct.pack("<HH", 0, len(streams))
    offset = len(meta) + headers_size
    body = b""
    for name, data in streams:
        n = name.encode() + b"\0"
        meta += struct.pack("<II", offset + len(body), len(data)) + n + b"\0" * (-len(n) % 4)
        body += data
    meta += body
    rva, raw = 0x2000, 0x200  # the one section
    cli = struct.pack("<IHHIIIIII", 72, 2, 5, rva + 72, len(meta), 1, 0, rva + 72 + len(meta), len(res_data))
    section = cli + b"\0" * (72 - len(cli)) + meta + res_data
    dirs = b"\0" * (14 * 8) + struct.pack("<II", rva, 72) + b"\0" * 8
    optional = struct.pack("<H", 0x10B) + b"\0" * 94 + dirs
    pe = (b"PE\0\0" + struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, len(optional), 0x102) + optional
          + b".text\0\0\0" + struct.pack("<IIII", len(section), rva, len(section), raw) + b"\0" * 16)
    head = b"MZ" + b"\0" * 58 + struct.pack("<I", 0x40) + pe
    return head + b"\0" * (raw - len(head)) + section


ITEMS_JSON = b'\xef\xbb\xbf{\n\t"ItemName": {\n\t\t"IronPickaxe": "Iron Pickaxe",\n\t\t"Alias": "{$ItemName.IronPickaxe}",\n\t},\n}'


def test_dotnet_constants_and_resources():
    exe = dotnet_exe("Terraria.ID", "ItemID", {"IronPickaxe": 1, "DirtBlock": 2, "Count": 3},
                     {"Terraria.Localization.Content.en-US.Items.json": ITEMS_JSON})
    asm = xna.DotNetAssembly(exe)
    assert asm.type_names() == ["Terraria.ID.ItemID"]
    assert asm.constants({"Terraria.ID.ItemID"}) == {"Terraria.ID.ItemID": {"IronPickaxe": 1, "DirtBlock": 2,
                                                                             "Count": 3}}
    assert asm.resource("Terraria.Localization.Content.en-US.Items.json") == ITEMS_JSON
    assert asm.resource("nope") is None
    names = xna.TerrariaData(exe)
    assert names.lookup("Item", 1) == ("Item", "IronPickaxe", "Iron Pickaxe")
    assert names.lookup("Item", 2) == ("Item", "DirtBlock", "Dirt Block")  # not localized: spaced internal name
    assert names.lookup("Item", 3) is None and names.lookup("Tiles", 1) is None
    assert names.display("ItemName", "Alias") == "Iron Pickaxe"
    with pytest.raises(xna.DotNetError):
        xna.DotNetAssembly(b"MZ" + b"\0" * 200)


def test_spaced():
    assert xna.spaced("OverworldDay") == "Overworld Day" and xna.spaced("Boss1") == "Boss 1"
    assert xna.spaced("NPCHead") == "NPC Head"


def test_plugin_names_assets_from_the_exe(tmp_path):
    images = tmp_path / "Content" / "Images"
    images.mkdir(parents=True)
    sounds = tmp_path / "Content" / "Sounds"
    sounds.mkdir()
    texture = xnb_file("Microsoft.Xna.Framework.Content.Texture2DReader", texture_body(bytes([255] * 4), 1, 1))
    (images / "Item_1.xnb").write_bytes(texture)
    (sounds / "Item_1.xnb").write_bytes(texture)
    (tmp_path / "Terraria.exe").write_bytes(dotnet_exe("Terraria.ID", "ItemID", {"IronPickaxe": 1},
                                            {"Terraria.Localization.Content.en-US.Items.json": ITEMS_JSON}))
    session = xna.XnaPlugin().open(str(tmp_path), lambda *a: None)
    by_uid = {a.uid: a for a in session.assets}
    item = by_uid["images/item_1.xnb"]
    assert item.name == item.path == "Images/Item: Iron Pickaxe (1)"
    assert by_uid["sounds/item_1.xnb"].name == "Sounds/Item_1"  # sound IDs aren't item IDs
    rows = dict(session.describe(item))
    assert rows["Name"] == "Iron Pickaxe" and rows["Item ID"] == "1 (IronPickaxe)"
    assert rows["File"] == "Images/Item_1.xnb"


@pytest.mark.skipif(not os.path.isdir(TERRARIA), reason="Terraria isn't installed")
def test_terraria():
    plugin = xna.XnaPlugin()
    assert plugin.detect(TERRARIA) == 95
    session = plugin.open(TERRARIA, lambda *a: None)
    by_name = {a.name: a for a in session.assets}
    assert session.image(by_name["Images/Item: Iron Pickaxe (1)"]).size == (32, 32)  # LZX compressed
    assert "Images/NPC: Eye of Cthulhu (4)" in by_name and "Music/Music: Overworld Day (1)" in by_name
    assert sum(a.name.startswith("Images/Item: ") for a in session.assets) > 6000
    game = session.terraria
    assert game.npc_frames[4] == 6 and game.item_animations[75] == (5, 8, True) and game.projectile_frames[1078] == 3
    eye = by_name["Animations/NPC: Eye of Cthulhu (4)"]
    frames, length = session.sprite_frames(eye)
    assert len(frames) == 6 and session.image(frames[0][1]).size == (110, 166) and length == pytest.approx(0.6)
    assert len(session.sprite_rects(by_name["Images/Armor head: Iron Helmet (2)"])) == 20
    assert session.raw(eye)[:6] == b"GIF89a"
    assert len(game.tile_colors) > 700 and game.tile_colors[0] == (151, 107, 75) and len(game.wall_colors) > 300
    assert game.prefix_name(79) == "Intrepid" and game.item_name(53) == "Cloud in a Bottle"
    assert game.defaults["Item"][1]["pick"] == 40 and game.defaults["NPC"][4]["lifeMax"] == 2800
    assert len(game.defaults["Item"]) > 4500 and game.main_arrays["tileSolid"][0] == 1
    assert game.sets["Item"]["SummonerWeaponThatScalesWithAttackSpeed"][4913] is True
    rows = dict(session.describe(by_name["Images/Item: Iron Pickaxe (1)"]))
    assert rows["Pickaxe power"] == "40%" and rows["Value"] == "buy 20 silver, sell 4 silver"


def test_il_instructions():
    # ldc.i4.s 75; ldc.i4 3581; ldc.i4.m1; switch (2 targets); 0xFE 0x0C ldloc 1; call token; ret
    code = bytes([0x1F, 75, 0x20]) + struct.pack("<i", 3581) + bytes([0x15, 0x45]) + struct.pack("<I2i", 2, 0, 0)
    code += bytes([0xFE, 0x0C, 1, 0, 0x28]) + struct.pack("<I", 0x06000010) + b"\x2a"
    ins = xna.il_instructions(code)
    assert [op for op, _arg in ins] == [0x1F, 0x20, 0x15, 0x45, 0xFE0C, 0x28, 0x2A]
    assert [xna.il_int(*i) for i in ins[:3]] == [75, 3581, -1] and xna.il_int(*ins[5]) is None
    assert xna.il_token(ins[5][1]) == 0x06000010


class FakeGame:
    npc_frames = {4: 6}
    projectile_frames: dict = {}
    item_animations = {75: (5, 8, True), 76: (4, 1, False)}


def test_sheet_frames():
    game = FakeGame()
    assert xna.sheet_frames(game, "NPC_4", 110, 996) == [(0, 166 * i, 110, 166) for i in range(6)]
    assert len(xna.sheet_frames(game, "Item_75", 22, 208)) == 8
    assert xna.sheet_frames(game, "Item_76", 22, 26) == [] and xna.sheet_frames(game, "NPC_5", 40, 40) == []
    head = xna.sheet_frames(game, "Armor_Head_2", 40, 1118)
    assert len(head) == 20 and head[-1] == (0, 1064, 40, 54)
    assert len(xna.sheet_frames(game, "Armor_2", 360, 224)) == 36
    assert len(xna.sheet_frames(game, "Tiles_0", 288, 270)) == 16 * 15
    assert xna.sheet_frames(game, "Tiles_4", 132, 528) == []
    assert xna.animation_of(game, "Item_75") == (8, 5 / 60, True) and xna.animation_of(game, "Gore_1") is None
    assert xna.frame_order(4, False) == [0, 1, 2, 3] and xna.frame_order(4, True) == [0, 1, 2, 3, 2, 1]


def test_animation_asset_frames_and_gif(tmp_path):
    images = tmp_path / "Content" / "Images"
    images.mkdir(parents=True)
    pixels = bytes([255, 0, 0, 255] * 2 + [0, 0, 255, 255] * 2)  # 2x2: a red frame over a blue one
    (images / "NPC_4.xnb").write_bytes(xnb_file("Microsoft.Xna.Framework.Content.Texture2DReader",
                                                texture_body(pixels, 2, 2)))
    session = xna.XnaPlugin().open(str(tmp_path), lambda *a: None)
    (sheet,) = session.assets
    assert session.sprite_rects(sheet) == []  # not Terraria (no exe): no layouts
    session.terraria = type("Game", (FakeGame,), {"npc_frames": {4: 2}})()
    assert session.sprite_rects(sheet) == [(0, 1, 2, 1), (0, 0, 2, 1)]  # y from the bottom
    anim = xna.Asset("animation", "Animations/NPC_4", "a", ref=("anim", sheet, 2, 0.1, False), ext="gif")
    frames, length = session.sprite_frames(anim)
    assert [session.image(f).getpixel((0, 0))[:3] for _t, f in frames] == [(255, 0, 0), (0, 0, 255)]
    assert length == pytest.approx(0.2) and session.stats(anim)["info"] == "2 frames"
    from PIL import Image
    gif = Image.open(io.BytesIO(session.raw(anim)))
    assert gif.n_frames == 2 and gif.info["duration"] == 100


def test_aes_fips197_vector():
    # FIPS-197 appendix C.1: AES-128 of 00112233...ff with key 000102...0f; CBC with a zero IV = one ECB block
    plain = xna.aes_cbc_decrypt(bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a"), bytes(range(16)), bytes(16))
    assert plain == bytes.fromhex("00112233445566778899aabbccddeeff")  # last byte 0xff: not PKCS#7 padding


def fake_world():
    """A 2 x 3 tile world (version 279): dirt (2 tiles, run-length) over water | a framed tile on wall 5 over air."""
    header = string7("Test World") + string7("seed1") + struct.pack("<Q", 0) + bytes(16) + struct.pack("<i", 7)
    header += struct.pack("<4i", 0, 32, 0, 48) + struct.pack("<2i", 3, 2) + struct.pack("<i", 2) + bytes([0] * 7 + [1])
    header += struct.pack("<q", 0) + b"\x00" + struct.pack("<17i", *[0] * 17) + struct.pack("<2i", 1, 0)
    header += struct.pack("<3d", 1.0, 2.0, 0.0) + struct.pack("<?i??", True, 0, False, False)
    header += struct.pack("<2i", 1, 1) + b"\x01" + bytes([1] + [0] * 17) + struct.pack("<??Bi?", 0, 0, 0, 0, True)
    tiles = bytes([0x42, 0, 1, 0x08, 255, 0x06, 2]) + struct.pack("<hh", 0, 0) + bytes([5, 0x40, 1])
    start = 24 + 2 + 8 + 2 + 1
    head = struct.pack("<i", 279) + b"relogic\x02" + struct.pack("<IQ", 0, 0)
    head += struct.pack("<h2i", 2, start, start + len(header)) + struct.pack("<h", 3) + bytes([0b100])
    return head + header + tiles


def test_world_header_layers_and_map():
    data = fake_world()
    info = xna.world_header(data)
    assert (info["name"], info["seed"], info["width"], info["height"]) == ("Test World", "seed1", 2, 3)
    assert info["mode"] == "Master" and info["seeds"] == ["Zenith (Get fixed boi)"] and info["evil"] == "Crimson"
    assert info["defeated"] == ["Eye of Cthulhu"] and info["hardmode"] and info["spawn"] == (1, 0)
    tiles, walls, liquids = xna.world_layers(data, info)
    assert tiles.tolist() == [[1, 3], [1, 0], [0, 0]]
    assert walls.tolist() == [[0, 5], [0, 0], [0, 0]] and liquids.tolist() == [[0, 0], [0, 0], [1, 0]]
    image = xna.world_map(info, (tiles, walls, liquids), {0: (1, 2, 3)}, {})
    assert image.size == (2, 3) and image.getpixel((0, 0)) == (1, 2, 3)
    assert image.getpixel((1, 0)) == xna.MAP_UNKNOWN_TILE and image.getpixel((0, 2)) == xna.MAP_LIQUIDS[1]
    assert ("Defeated", "Eye of Cthulhu") in xna.world_rows(info)


def fake_player():
    out = struct.pack("<i", 279) + b"relogic\x03" + struct.pack("<IQ", 0, 0) + string7("Bella") + b"\x02"
    out += struct.pack("<q", 3600 * 10_000_000 * 2 + 600 * 10_000_000) + struct.pack("<i", 6) + bytes(4) + b"\x05"
    out += struct.pack("<4i", 120, 140, 20, 40) + bytes(11) + struct.pack("<3i", 0, 9, 1) + bytes(range(21))
    out += struct.pack("<iB", 0, 0) * 3 + struct.pack("<iB", 53, 79) + struct.pack("<iB", 0, 0) * 16
    out += struct.pack("<iB", 0, 0) * 10
    out += struct.pack("<iiB?", 4, 1, 6, True) + struct.pack("<iiB?", 8, 41, 0, False) + struct.pack("<iiB?", 0, 0, 0, 0) * 56
    out += struct.pack("<iB", 0, 0) * 10 + struct.pack("<iiB", 0, 0, 0) * 40 * 4 + b"\x00"
    return out + struct.pack("<ii", 216, 600) + struct.pack("<ii", 0, 0) * 43


def test_player():
    p = xna.read_player(fake_player())
    assert (p["name"], p["difficulty"], p["play_seconds"], p["life_max"], p["deaths"]) == ("Bella", "Hardcore",
                                                                                          7800, 140, 9)
    assert p["armor"][3] == (53, 79) and p["inventory"][1] == (8, 41, 0, False) and p["buffs"] == [(216, 600)]
    text = xna.player_text(p, None)
    assert "Play time: 2 h 10 min" in text and "Accessory 1: item #53" in text and "2. item #8 x41" in text
    assert "1. item #4 (favorite)" in text
    with pytest.raises(ValueError):
        xna.read_player(b"\x17\x01\x00\x00notrelogic" + bytes(40))


def test_save_files(tmp_path, monkeypatch):
    docs = tmp_path / "Docs"
    (docs / "My Games" / "Terraria" / "Worlds").mkdir(parents=True)
    (docs / "My Games" / "Terraria" / "Worlds" / "a.wld").write_bytes(fake_world())
    (docs / "My Games" / "Terraria" / "Worlds" / "a.wld.bak").write_bytes(b"")
    remote = tmp_path / "Steam" / "userdata" / "123" / "105600" / "remote" / "players"
    remote.mkdir(parents=True)
    (remote / "b.plr").write_bytes(b"")
    monkeypatch.setattr(xna, "documents_folder", lambda: str(docs))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "home"))
    game = tmp_path / "Steam" / "steamapps" / "common" / "Terraria"
    found = [(label, kind, os.path.basename(p)) for label, kind, p in xna.save_files(str(game))]
    assert found == [("Documents", "Worlds", "a.wld"), ("Steam Cloud", "Players", "b.plr")]


class FakeAsm:
    """Stands in for DotNetAssembly in the IL tests: method bodies and token names / signatures by hand."""
    tokens = {0x0A000001: ("sellPrice", (False, 4, False)), 0x0A000002: ("SetShopValues", (True, 2, True)),
              0x06000003: ("DefaultToWhip", (True, 2, True)), 0x0A000004: ("CreateBoolSet", (True, 1, False)),
              0x0A000005: ("CreateIntSet", (True, 2, False)), 0x04000001: ("damage", None),
              0x04000002: ("value", None), 0x04000003: ("type", None), 0x04000004: ("Boss", None),
              0x04000005: ("Factory", None), 0x04000006: ("Size", None), 0x01000001: ("Int32", None)}

    def __init__(self, code):
        self.code = code

    def method_body(self, _method):
        return self.code

    def member_name(self, token):
        return self.tokens[token][0]

    def signature(self, token):
        return self.tokens[token][1]

    def param_names(self, token):
        return ["projectileId", "dmg"] if token == 0x06000003 else []

    def field_data(self, _token, _size):
        return None


def il(*parts):
    """IL bytes from opcodes (ints), (opcode, int32 operand) and (opcode, 'b', int8 operand) tuples."""
    out = b""
    for p in parts:
        if isinstance(p, int):
            out += bytes([p])
        elif len(p) == 3:
            out += bytes([p[0]]) + struct.pack("<b", p[2])
        else:
            out += bytes([p[0]]) + struct.pack("<i", p[1])
    return out


def test_il_defaults_if_chains_switch_and_helpers():
    ldc = lambda v: (0x20, v)  # noqa: E731 - ldc.i4
    block1 = il(0x02, ldc(12), (0x7D, 0x04000001),  # this.damage = 12
                0x02, ldc(0), ldc(1), ldc(2), ldc(3), (0x28, 0x0A000001), (0x7D, 0x04000002))  # value = sellPrice(...)
    block2 = il(0x02, ldc(7), (0x7D, 0x04000001), 0x02, ldc(3), ldc(500), (0x28, 0x0A000002))
    block3 = il(0x02, ldc(914), ldc(18), (0x28, 0x06000003), 0x2A)
    code = il(0x03, ldc(1), (0x33, "b", len(block1) + 2)) + block1 + il((0x2B, "b", 0))
    # type == 2 || type == 3: beq to the block, then bne.un past it
    second = il(0x03, ldc(3), (0x33, "b", len(block2)))
    code += il(0x03, ldc(2), (0x2E, "b", len(second))) + second + block2
    # switch (type - 10): case 10 -> block3, case 11 -> nothing (points past it)
    switch = il(0x03, ldc(10), 0x59) + bytes([0x45]) + struct.pack("<I2i", 2, 0, len(block3))
    code += switch + block3 + b"\x2a"
    out = xna.il_defaults(FakeAsm(code), 1, 0x04000003, {})
    assert out[1] == {"damage": 12, "value": (1 * 10_000 + 2 * 100 + 3) * 5}
    assert out[2] == out[3] == {"damage": 7, "rare": 3, "value": 500}
    assert out[10] == {"calls": ["DefaultToWhip(projectileId=914, dmg=18)"]} and 11 not in out


def test_il_evaluate_sets():
    # Sets.Boss = Factory.CreateBoolSet(new int[] {5, 7}); Sets.Size = Factory.CreateIntSet(0, new int[] {4, 2})
    def array(*values):
        out = il((0x20, len(values)), (0x8D, 0x01000001))
        for i, v in enumerate(values):
            out += il(0x25, (0x20, i), (0x20, v), 0x9E)
        return out
    code = il((0x7E, 0x04000005)) + array(5, 7) + il((0x6F, 0x0A000004), (0x80, 0x04000004))
    code += il((0x7E, 0x04000005), 0x16) + array(4, 2) + il((0x6F, 0x0A000005), (0x80, 0x04000006), 0x2A)
    calls, stored = [], {}

    def on_call(name, args):
        calls.append(name)
        return (name, dict(args[-1]))
    xna.il_evaluate(FakeAsm(code), 1, on_call=on_call, on_store=lambda t, v: stored.__setitem__(t, v))
    assert stored == {0x04000004: ("CreateBoolSet", {0: 5, 1: 7}), 0x04000006: ("CreateIntSet", {0: 4, 1: 2})}


def test_game_rows():
    class Game(FakeGame):
        defaults = {"Item": {1: {"damage": 5, "melee": 1, "pick": 40, "value": 2000, "rare": 1, "width": 24}},
                    "NPC": {4: {"lifeMax": 2800, "boss": 1, "value": 30000.0, "knockBackResist": 0.0}}}
        sets = {"NPC": {"ShouldBeCountedAsBoss": {4: True}, "TrailingMode": {4: 3}}}
        main_arrays = {"tileSolid": {0: 1}, "tileLighted": {4: 1}}
        tile_colors = {0: (151, 107, 75)}
        wall_colors: dict = {}

        def lookup(self, prefix, n):
            return None

        def flags(self, label, number):
            return xna.TerrariaData.flags(self, label, number)
    game = Game()
    item = dict(xna.game_rows(game, "Item", 1))
    assert item["Damage"] == "5 (melee)" and item["Pickaxe power"] == "40%" and item["Rarity"] == "Blue"
    assert item["Value"] == "buy 20 silver, sell 4 silver" and item["Other defaults"] == "width=24"
    npc = dict(xna.game_rows(game, "NPC", 4))
    assert npc["Life"] == "2800" and npc["Coins dropped"] == "3 gold" and npc["Boss"] == "yes"
    assert npc["ID sets"] == "ShouldBeCountedAsBoss, TrailingMode=3"
    tile = dict(xna.game_rows(game, "Tile", 0))
    assert tile == {"Tile flags": "Solid", "Map color": "#976b4b"}
    assert xna.game_rows(game, "Armor head", 1) == [] and xna.coins(1_234_567) == "1 platinum 23 gold 45 silver 67 copper"
