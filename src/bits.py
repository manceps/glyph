"""Bit manipulation and CRC utilities for Glyph encoding.

Handles conversion between hex strings, byte arrays, and bit sequences.
Includes CRC-32 computation for strong error detection.
"""

from __future__ import annotations

import zlib

# Glyph type constants (extensible -- applications can define their own)
GLYPH_TYPE_MEMBER = 0x01
GLYPH_TYPE_COMMUNITY = 0x02
GLYPH_TYPE_INVITE = 0x03
GLYPH_TYPE_RECOVERY = 0x04

GLYPH_TYPE_NAMES: dict[int, str] = {
    GLYPH_TYPE_MEMBER: "MEMBER",
    GLYPH_TYPE_COMMUNITY: "COMMUNITY",
    GLYPH_TYPE_INVITE: "INVITE",
    GLYPH_TYPE_RECOVERY: "RECOVERY",
}

GLYPH_TYPE_BY_NAME: dict[str, int] = {v: k for k, v in GLYPH_TYPE_NAMES.items()}


def hex_to_bits(hex_string: str) -> list[int]:
    """Convert a hex string to a list of bits (MSB first).

    Args:
        hex_string: Hex string (e.g. "a3f0"). May have 0x prefix.

    Returns:
        List of 0s and 1s, MSB first.

    Raises:
        ValueError: If hex_string is not valid hex.
    """
    clean = hex_string.removeprefix("0x").removeprefix("0X")
    if not clean:
        raise ValueError("Empty hex string")
    data = bytes.fromhex(clean)
    bits: list[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_hex(bits: list[int]) -> str:
    """Convert a list of bits back to a hex string.

    Args:
        bits: List of 0s and 1s, MSB first. Padded to byte boundary.

    Returns:
        Lowercase hex string without 0x prefix.
    """
    # Pad to byte boundary
    padded = list(bits)
    while len(padded) % 8 != 0:
        padded.append(0)

    result = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        result.append(byte)
    return result.hex()


def compute_crc32(data: bytes) -> int:
    """Compute CRC-32 checksum.

    Args:
        data: Input bytes.

    Returns:
        CRC-32 value (0-4294967295).
    """
    return zlib.crc32(data) & 0xFFFFFFFF


def append_crc(hex_data: str) -> str:
    """Append CRC-32 (4 bytes) to hex data.

    Args:
        hex_data: Hex string of payload data.

    Returns:
        Hex string with 4-byte CRC appended.
    """
    clean = hex_data.removeprefix("0x").removeprefix("0X")
    data = bytes.fromhex(clean)
    crc = compute_crc32(data)
    return clean + f"{crc:08x}"


def validate_crc(hex_data_with_crc: str) -> bool:
    """Validate CRC-32 on hex data (last 4 bytes are CRC).

    Args:
        hex_data_with_crc: Hex string where last 4 bytes are CRC-32.

    Returns:
        True if CRC is valid.
    """
    clean = hex_data_with_crc.removeprefix("0x").removeprefix("0X")
    if len(clean) < 10:  # minimum: 1 data byte + 4 CRC bytes = 10 hex chars
        return False

    # Split data and CRC
    data_hex = clean[:-8]
    crc_hex = clean[-8:]

    try:
        data = bytes.fromhex(data_hex)
        expected_crc = int(crc_hex, 16)
        return compute_crc32(data) == expected_crc
    except ValueError:
        return False
