"""Tests for bit manipulation and CRC utilities."""

import pytest

from src.bits import append_crc, bits_to_hex, compute_crc32, hex_to_bits, validate_crc


class TestHexToBits:
    def test_single_byte(self):
        bits = hex_to_bits("a3")
        assert bits == [1, 0, 1, 0, 0, 0, 1, 1]

    def test_two_bytes(self):
        bits = hex_to_bits("ff00")
        assert bits == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_with_0x_prefix(self):
        bits = hex_to_bits("0xa3")
        assert bits == [1, 0, 1, 0, 0, 0, 1, 1]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty hex string"):
            hex_to_bits("")

    def test_invalid_hex_raises(self):
        with pytest.raises(ValueError):
            hex_to_bits("zz")


class TestBitsToHex:
    def test_single_byte(self):
        assert bits_to_hex([1, 0, 1, 0, 0, 0, 1, 1]) == "a3"

    def test_roundtrip(self):
        original = "deadbeef"
        bits = hex_to_bits(original)
        result = bits_to_hex(bits)
        assert result == original

    def test_padding(self):
        # 4 bits -> padded to 8
        result = bits_to_hex([1, 0, 1, 0])
        assert result == "a0"


class TestCRC32:
    def test_known_value(self):
        crc = compute_crc32(b"\x00")
        assert 0 <= crc <= 0xFFFFFFFF

    def test_different_data_different_crc(self):
        crc1 = compute_crc32(b"\x00")
        crc2 = compute_crc32(b"\x01")
        assert crc1 != crc2

    def test_all_zeros_nonzero_crc(self):
        """CRC-32 of all-zeros must NOT be zero."""
        crc = compute_crc32(b"\x00" * 32)
        assert crc != 0

    def test_append_and_validate(self):
        data = "deadbeef"
        with_crc = append_crc(data)
        assert len(with_crc) == len(data) + 8  # 4 CRC bytes = 8 hex chars
        assert validate_crc(with_crc)

    def test_validate_corrupted_fails(self):
        data = "deadbeef"
        with_crc = append_crc(data)
        # Corrupt a byte in data, keep original CRC
        corrupted = "aeadbeef" + with_crc[-8:]
        assert not validate_crc(corrupted)

    def test_validate_too_short(self):
        assert not validate_crc("ab")  # Only 1 byte, need at least 5
        assert not validate_crc("abcd1234")  # Only 4 bytes, need data + 4 CRC

    def test_append_with_0x_prefix(self):
        with_crc = append_crc("0xdead")
        assert validate_crc(with_crc)


class TestGlyphTypeConstants:
    def test_type_values_are_distinct(self):
        from src.bits import (
            GLYPH_TYPE_COMMUNITY,
            GLYPH_TYPE_INVITE,
            GLYPH_TYPE_MEMBER,
            GLYPH_TYPE_RECOVERY,
        )

        types = [GLYPH_TYPE_MEMBER, GLYPH_TYPE_COMMUNITY, GLYPH_TYPE_INVITE, GLYPH_TYPE_RECOVERY]
        assert len(types) == len(set(types)), "Glyph type values must be unique"

    def test_type_values_are_single_byte(self):
        from src.bits import (
            GLYPH_TYPE_COMMUNITY,
            GLYPH_TYPE_INVITE,
            GLYPH_TYPE_MEMBER,
            GLYPH_TYPE_RECOVERY,
        )

        for t in [GLYPH_TYPE_MEMBER, GLYPH_TYPE_COMMUNITY, GLYPH_TYPE_INVITE, GLYPH_TYPE_RECOVERY]:
            assert 0x00 < t <= 0xFF, f"Type {t:#x} must fit in a single byte and be nonzero"

    def test_type_names_mapping(self):
        from src.bits import GLYPH_TYPE_BY_NAME, GLYPH_TYPE_NAMES

        assert GLYPH_TYPE_NAMES[0x01] == "MEMBER"
        assert GLYPH_TYPE_NAMES[0x02] == "COMMUNITY"
        assert GLYPH_TYPE_NAMES[0x03] == "INVITE"
        assert GLYPH_TYPE_NAMES[0x04] == "RECOVERY"

        assert GLYPH_TYPE_BY_NAME["MEMBER"] == 0x01
        assert GLYPH_TYPE_BY_NAME["COMMUNITY"] == 0x02

    def test_type_names_roundtrip(self):
        from src.bits import GLYPH_TYPE_BY_NAME, GLYPH_TYPE_NAMES

        for code, name in GLYPH_TYPE_NAMES.items():
            assert GLYPH_TYPE_BY_NAME[name] == code
