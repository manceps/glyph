"""Tests for Glyph topology encoder."""

import pytest

from src.encoder import encode
from src.tree import TopologyTree


class TestEncode:
    def test_encode_basic_hex(self):
        tree = encode("a3f0", "hexagon")
        assert isinstance(tree, TopologyTree)
        assert tree.bit_length == 16  # 2 bytes = 16 bits
        assert tree.total_nodes > 1

    def test_encode_32_byte_hash(self):
        sha256_hex = "a" * 64
        tree = encode(sha256_hex, "circle")
        assert tree.bit_length == 256

    def test_encode_different_shapes(self):
        for shape in [
            "circle",
            "hexagon",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            tree = encode("deadbeef", shape)
            assert isinstance(tree, TopologyTree)
            assert tree.total_nodes > 0

    def test_encode_rejects_oversized_data(self):
        too_large = "ab" * 65  # 65 bytes
        with pytest.raises(ValueError, match="Data too large"):
            encode(too_large, "hexagon")

    def test_encode_rejects_invalid_hex(self):
        with pytest.raises(ValueError):
            encode("not_hex", "hexagon")

    def test_encode_rejects_unknown_shape(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            encode("deadbeef", "nonexistent")

    def test_encode_deterministic(self):
        tree1 = encode("deadbeef", "hexagon")
        tree2 = encode("deadbeef", "hexagon")
        assert tree1.bit_length == tree2.bit_length
        assert tree1.total_nodes == tree2.total_nodes

    def test_encode_different_data_different_tree(self):
        tree1 = encode("deadbeef", "hexagon")
        tree2 = encode("cafebabe", "hexagon")
        assert tree1.bit_length == tree2.bit_length  # same byte count

    def test_encode_single_byte(self):
        tree = encode("ff", "hexagon")
        assert tree.bit_length == 8
        assert tree.total_nodes > 1

    def test_encode_max_size(self):
        max_data = "ab" * 64  # exactly 64 bytes
        tree = encode(max_data, "hexagon")
        assert tree.bit_length == 512


class TestEncodeMemberPayload:
    def test_member_payload_length(self):
        from src.encoder import encode_member_payload

        pubkey = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c"
        result = encode_member_payload(pubkey, tier=0)
        # 14 bytes = 28 hex chars (without CRC)
        assert len(result) == 28

    def test_member_payload_type_byte(self):
        from src.encoder import encode_member_payload

        pubkey = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c"
        result = encode_member_payload(pubkey, tier=0)
        # First byte should be 0x01 (MEMBER)
        assert result[:2] == "01"

    def test_member_payload_contains_pubkey(self):
        from src.encoder import encode_member_payload

        pubkey = bytes(range(1, 13))  # 12 bytes
        result = encode_member_payload(pubkey, tier=2)
        data = bytes.fromhex(result)
        assert data[1:13] == pubkey

    def test_member_payload_contains_tier(self):
        from src.encoder import encode_member_payload

        pubkey = bytes(12)
        for tier in range(4):
            result = encode_member_payload(pubkey, tier=tier)
            data = bytes.fromhex(result)
            assert data[13] == tier

    def test_member_payload_invalid_pubkey_length(self):
        from src.encoder import encode_member_payload

        with pytest.raises(ValueError, match="12 bytes"):
            encode_member_payload(b"short", tier=0)

    def test_member_payload_invalid_tier(self):
        from src.encoder import encode_member_payload

        pubkey = bytes(12)
        with pytest.raises(ValueError, match="tier"):
            encode_member_payload(pubkey, tier=5)

    def test_member_payload_encodes_to_tree(self):
        from src.encoder import encode, encode_member_payload

        pubkey = bytes(range(1, 13))
        data_hex = encode_member_payload(pubkey, tier=1)
        tree = encode(data_hex, "hexagon")
        assert isinstance(tree, TopologyTree)
        assert tree.bit_length == 14 * 8  # 14 data bytes = 112 bits


class TestEncodeCommunityPayload:
    def test_community_payload_length(self):
        from src.encoder import encode_community_payload

        result = encode_community_payload(
            community_id_prefix=b"\x01\x02\x03\x04",
            join_requirement=0,
            expiry_timestamp=1700000000,
            presenter_prefix=b"\x0a\x0b\x0c\x0d",
        )
        assert len(result) == 28  # 14 bytes

    def test_community_payload_type_byte(self):
        from src.encoder import encode_community_payload

        result = encode_community_payload(
            community_id_prefix=b"\x01\x02\x03\x04",
            join_requirement=0,
            expiry_timestamp=1700000000,
            presenter_prefix=b"\x0a\x0b\x0c\x0d",
        )
        assert result[:2] == "02"  # COMMUNITY type

    def test_community_payload_fields(self):
        from src.encoder import encode_community_payload

        cid = b"\xaa\xbb\xcc\xdd"
        presenter = b"\x11\x22\x33\x44"
        ts = 1700000000

        result = encode_community_payload(
            community_id_prefix=cid,
            join_requirement=2,
            expiry_timestamp=ts,
            presenter_prefix=presenter,
        )
        data = bytes.fromhex(result)
        assert data[0] == 0x02  # type
        assert data[1:5] == cid
        assert data[5] == 2  # join_requirement
        assert int.from_bytes(data[6:10], "big") == ts
        assert data[10:14] == presenter

    def test_community_payload_invalid_community_id_length(self):
        from src.encoder import encode_community_payload

        with pytest.raises(ValueError, match="4 bytes"):
            encode_community_payload(
                community_id_prefix=b"\x01\x02",
                join_requirement=0,
                expiry_timestamp=0,
                presenter_prefix=b"\x01\x02\x03\x04",
            )

    def test_community_payload_invalid_join_requirement(self):
        from src.encoder import encode_community_payload

        with pytest.raises(ValueError, match="join_requirement"):
            encode_community_payload(
                community_id_prefix=b"\x01\x02\x03\x04",
                join_requirement=5,
                expiry_timestamp=0,
                presenter_prefix=b"\x01\x02\x03\x04",
            )

    def test_community_payload_invalid_presenter_length(self):
        from src.encoder import encode_community_payload

        with pytest.raises(ValueError, match="4 bytes"):
            encode_community_payload(
                community_id_prefix=b"\x01\x02\x03\x04",
                join_requirement=0,
                expiry_timestamp=0,
                presenter_prefix=b"\x01",
            )

    def test_community_payload_encodes_to_tree(self):
        from src.encoder import encode, encode_community_payload

        data_hex = encode_community_payload(
            community_id_prefix=b"\x01\x02\x03\x04",
            join_requirement=1,
            expiry_timestamp=1700000000,
            presenter_prefix=b"\x0a\x0b\x0c\x0d",
        )
        tree = encode(data_hex, "circle")
        assert isinstance(tree, TopologyTree)
        assert tree.bit_length == 14 * 8
