"""AES-256 ECB decryption in numpy (what Unreal uses for encrypted .pak / IoStore files).

No crypto package needed: all 16-byte blocks are decrypted at once with table lookups, which is
fast enough for pak indexes and single assets.
"""

import numpy as np


def _tables():
    sbox = np.zeros(256, np.uint8)
    p = q = 1
    sbox[0] = 0x63
    while True:
        # p *= 3 ; q /= 3 in GF(2^8)
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1) | (q >> 7)) & 0xFF ^ ((q << 2) | (q >> 6)) & 0xFF ^ ((q << 3) | (q >> 5)) & 0xFF \
            ^ ((q << 4) | (q >> 4)) & 0xFF
        sbox[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    inv = np.zeros(256, np.uint8)
    inv[sbox] = np.arange(256, dtype=np.uint8)
    return sbox, inv


def _gmul(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        a = ((a << 1) ^ (0x1B if a & 0x80 else 0)) & 0xFF
        b >>= 1
    return out


SBOX, INV_SBOX = _tables()
MUL = {n: np.array([_gmul(i, n) for i in range(256)], np.uint8) for n in (9, 11, 13, 14)}
# InvShiftRows on a column-major 16-byte state
INV_SHIFT = np.array([0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3])


def expand_key(key):
    """AES-256 round keys: (15, 16) uint8."""
    if len(key) != 32:
        raise ValueError("AES-256 needs a 32-byte (64 hex digit) key")
    words = [list(key[i:i + 4]) for i in range(0, 32, 4)]
    rcon = 1
    for i in range(8, 60):
        t = list(words[i - 1])
        if i % 8 == 0:
            t = t[1:] + t[:1]
            t = [int(SBOX[b]) for b in t]
            t[0] ^= rcon
            rcon = _gmul(rcon, 2)
        elif i % 8 == 4:
            t = [int(SBOX[b]) for b in t]
        words.append([a ^ b for a, b in zip(words[i - 8], t)])
    return np.array([sum(words[r * 4:r * 4 + 4], []) for r in range(15)], np.uint8)


def parse_key(text):
    """'0x1234...' / '1234...' (64 hex digits) or base64 -> 32 bytes."""
    text = text.strip()
    if text.lower().startswith("0x"):
        text = text[2:]
    try:
        key = bytes.fromhex(text)
    except ValueError:
        import base64
        key = base64.b64decode(text)
    if len(key) != 32:
        raise ValueError("An Unreal AES key is 32 bytes (64 hex digits)")
    return key


class AES:
    def __init__(self, key):
        self.round_keys = expand_key(key)

    def decrypt(self, data):
        """Decrypt ECB; len(data) must be a multiple of 16."""
        if len(data) % 16:
            raise ValueError("AES data must be a multiple of 16 bytes")
        if not data:
            return b""
        s = np.frombuffer(data, np.uint8).reshape(-1, 16).copy()
        rk = self.round_keys
        s ^= rk[14]
        for rnd in range(13, 0, -1):
            s = INV_SBOX[s[:, INV_SHIFT]]
            s ^= rk[rnd]
            c = s.reshape(-1, 4, 4)
            a0, a1, a2, a3 = c[:, :, 0], c[:, :, 1], c[:, :, 2], c[:, :, 3]
            m9, m11, m13, m14 = MUL[9], MUL[11], MUL[13], MUL[14]
            s = np.stack([
                m14[a0] ^ m11[a1] ^ m13[a2] ^ m9[a3],
                m9[a0] ^ m14[a1] ^ m11[a2] ^ m13[a3],
                m13[a0] ^ m9[a1] ^ m14[a2] ^ m11[a3],
                m11[a0] ^ m13[a1] ^ m9[a2] ^ m14[a3],
            ], axis=2).reshape(-1, 16)
        s = INV_SBOX[s[:, INV_SHIFT]]
        s ^= rk[0]
        return s.tobytes()
