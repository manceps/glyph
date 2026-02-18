"""Tests for Glyph topology decoder."""

import io

from PIL import Image

from src.decoder import DecodeResult, decode_image
from src.encoder import encode
from src.renderer import render_png


class TestDecode:
    def test_decode_roundtrip(self):
        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        png_bytes = render_png(tree, "hexagon", size=512)

        result = decode_image(png_bytes)
        assert isinstance(result, DecodeResult)
        assert result.confidence == 0.95
        assert result.data_hex == data_hex
        assert result.error is None

    def test_decode_roundtrip_16_bytes(self):
        data_hex = "a3f0b1c2d4e5f6789012345678abcdef"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        png_bytes = render_png(tree, "hexagon", size=512)

        result = decode_image(png_bytes)
        assert result.confidence == 0.95
        assert result.data_hex == data_hex

    def test_decode_roundtrip_all_shapes(self):
        data_hex = "deadbeefcafebabe0011223344556677"  # pragma: allowlist secret
        for shape in [
            "hexagon",
            "circle",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            tree = encode(data_hex, shape)
            png_bytes = render_png(tree, shape, size=512)
            result = decode_image(png_bytes)
            assert result.confidence == 0.95, f"Failed for shape {shape}"
            assert result.data_hex == data_hex, f"Wrong data for shape {shape}"

    def test_decode_invalid_image(self):
        result = decode_image(b"not an image")
        assert result.data_hex is None
        assert result.confidence == 0.0
        assert result.error is not None

    def test_decode_empty_image(self):
        result = decode_image(b"")
        assert result.data_hex is None
        assert result.confidence == 0.0

    def test_decode_result_fields(self):
        result = DecodeResult(data_hex="abcd", confidence=0.95)
        assert result.data_hex == "abcd"
        assert result.confidence == 0.95
        assert result.error is None

    def test_decode_result_with_error(self):
        result = DecodeResult(data_hex=None, confidence=0.0, error="test error")
        assert result.error == "test error"

    def test_decode_glyph_on_white_background(self):
        """Simulate camera capture: glyph on white bg, surrounded by white."""
        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        glyph_png = render_png(tree, "hexagon", size=300)

        glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")
        bg = Image.new("RGBA", (800, 800), (255, 255, 255, 255))
        offset = (800 - 300) // 2
        bg.alpha_composite(glyph_img, dest=(offset, offset))

        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        photo_bytes = buf.getvalue()

        result = decode_image(photo_bytes)
        assert result.confidence == 0.95, f"Failed: {result.error}"
        assert result.data_hex == data_hex

    def test_decode_glyph_small_in_large_white_image(self):
        """Glyph is small relative to the image (like a cropped camera photo)."""
        data_hex = "deadbeefcafebabe0011223344556677"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        glyph_png = render_png(tree, "hexagon", size=200)

        glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")
        bg = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        offset = (1024 - 200) // 2
        bg.alpha_composite(glyph_img, dest=(offset, offset))

        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        photo_bytes = buf.getvalue()

        result = decode_image(photo_bytes)
        assert result.confidence == 0.95, f"Failed: {result.error}"
        assert result.data_hex == data_hex

    def test_decode_glyph_all_shapes_on_white_background(self):
        """All shapes decode correctly when on a white background."""
        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        for shape in [
            "hexagon",
            "circle",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            tree = encode(data_hex, shape)
            glyph_png = render_png(tree, shape, size=256)

            glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")
            bg = Image.new("RGBA", (600, 600), (255, 255, 255, 255))
            offset = (600 - 256) // 2
            bg.alpha_composite(glyph_img, dest=(offset, offset))

            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            photo_bytes = buf.getvalue()

            result = decode_image(photo_bytes)
            assert result.confidence == 0.95, f"Failed for shape {shape}: {result.error}"
            assert result.data_hex == data_hex, f"Wrong data for shape {shape}"

    def test_decode_glyph_white_card_on_dark_background(self):
        """Simulate real camera scan: glyph on white card, dark page around it."""
        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        glyph_png = render_png(tree, "hexagon", size=256)

        glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")

        # Dark page background
        frame = Image.new("RGBA", (1024, 1024), (15, 23, 42, 255))

        # White card in the upper-center area
        card_w, card_h = 350, 350
        card_x = (1024 - card_w) // 2
        card_y = 200
        white_card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
        frame.paste(white_card, (card_x, card_y))

        # Place glyph centered on the white card
        glyph_x = card_x + (card_w - 256) // 2
        glyph_y = card_y + (card_h - 256) // 2
        frame.alpha_composite(glyph_img, dest=(glyph_x, glyph_y))

        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        photo_bytes = buf.getvalue()

        result = decode_image(photo_bytes)
        assert result.confidence == 0.95, f"Failed: {result.error}"
        assert result.data_hex == data_hex

    def test_decode_glyph_white_card_dark_bg_all_shapes(self):
        """All shapes decode when on white card with dark background around it."""
        data_hex = "deadbeefcafebabe0011223344556677"  # pragma: allowlist secret
        for shape in [
            "hexagon",
            "circle",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            tree = encode(data_hex, shape)
            glyph_png = render_png(tree, shape, size=256)

            glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")
            frame = Image.new("RGBA", (800, 800), (15, 23, 42, 255))

            card_w, card_h = 320, 320
            card_x = (800 - card_w) // 2
            card_y = 150
            white_card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
            frame.paste(white_card, (card_x, card_y))

            glyph_x = card_x + (card_w - 256) // 2
            glyph_y = card_y + (card_h - 256) // 2
            frame.alpha_composite(glyph_img, dest=(glyph_x, glyph_y))

            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            photo_bytes = buf.getvalue()

            result = decode_image(photo_bytes)
            assert result.confidence == 0.95, f"Failed for {shape}: {result.error}"
            assert result.data_hex == data_hex, f"Wrong data for {shape}"

    def test_decode_jpeg_dark_bg_all_shapes(self):
        """All shapes decode from JPEG (what phone cameras typically send)."""
        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        for shape in [
            "hexagon",
            "circle",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            tree = encode(data_hex, shape)
            glyph_png = render_png(tree, shape, size=256)

            glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")
            frame = Image.new("RGBA", (512, 512), (15, 23, 42, 255))

            card_w, card_h = 300, 300
            card_x = (512 - card_w) // 2
            card_y = 50
            white_card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
            frame.paste(white_card, (card_x, card_y))

            glyph_x = card_x + (card_w - 256) // 2
            glyph_y = card_y + (card_h - 256) // 2
            frame.alpha_composite(glyph_img, dest=(glyph_x, glyph_y))

            buf = io.BytesIO()
            frame.convert("RGB").save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()

            result = decode_image(jpeg_bytes)
            assert result.confidence == 0.95, f"JPEG failed for {shape}: {result.error}"
            assert result.data_hex == data_hex, f"JPEG wrong data for {shape}"

    def test_decode_with_all_color_palettes(self):
        """All color palettes decode correctly (regression for light colors)."""
        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        palettes = [
            ["#1A365D", "#2B6CB0", "#4299E1", "#90CDF4", "#EBF8FF"],  # Blue
            ["#22543D", "#276749", "#48BB78", "#9AE6B4", "#F0FFF4"],  # Green
            ["#742A2A", "#C53030", "#FC8181", "#FEB2B2", "#FFF5F5"],  # Red
            ["#44337A", "#6B46C1", "#B794F4", "#D6BCFA", "#FAF5FF"],  # Purple
            ["#7B341E", "#C05621", "#F6AD55", "#FEEBC8", "#FFFAF0"],  # Orange
            ["#234E52", "#2C7A7B", "#4FD1C5", "#B2F5EA", "#E6FFFA"],  # Teal
            ["#1A202C", "#4A5568", "#A0AEC0", "#E2E8F0", "#F7FAFC"],  # Gray
            ["#5A3825", "#8B6914", "#D4A843", "#F0D68A", "#FFFBE6"],  # Gold
        ]
        for i, colors in enumerate(palettes):
            tree = encode(data_hex, "hexagon")
            png_bytes = render_png(tree, "hexagon", colors, size=512)
            result = decode_image(png_bytes)
            assert result.confidence == 0.95, f"Palette {i} failed: {result.error}"
            assert result.data_hex == data_hex, f"Palette {i} wrong: {result.data_hex}"

    def test_reject_dark_image_no_glyph(self):
        """Dark image with no glyph should return error, not all-FFs."""
        dark = Image.new("RGBA", (512, 512), (15, 23, 42, 255))
        buf = io.BytesIO()
        dark.convert("RGB").save(buf, format="JPEG", quality=85)
        result = decode_image(buf.getvalue())
        assert result.data_hex is None
        assert result.confidence == 0.0
        assert result.error is not None
        assert "dark" in result.error.lower() or "no" in result.error.lower()

    def test_reject_dark_image_tiny_white_speck(self):
        """Dark image with tiny white area should return error."""
        frame = Image.new("RGBA", (512, 512), (15, 23, 42, 255))
        speck = Image.new("RGBA", (30, 30), (255, 255, 255, 255))
        frame.paste(speck, (241, 241))
        buf = io.BytesIO()
        frame.convert("RGB").save(buf, format="JPEG", quality=85)
        result = decode_image(buf.getvalue())
        assert result.data_hex is None
        assert result.confidence == 0.0

    def test_decode_glyph_fills_most_of_frame(self):
        """Glyph filling >70% of image should still decode correctly."""
        data_hex = "a3f04b2c9e1d7f8a1234567890abcdef"  # pragma: allowlist secret
        tree = encode(data_hex, "circle")
        glyph_png = render_png(tree, "circle", size=256)
        glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")

        bg = Image.new("RGBA", (320, 320), (255, 255, 255, 255))
        bg.alpha_composite(glyph_img, dest=(32, 32))
        buf = io.BytesIO()
        bg.convert("RGB").save(buf, format="JPEG", quality=85)

        result = decode_image(buf.getvalue())
        assert result.confidence == 0.95, f"Failed: {result.error}"
        assert result.data_hex == data_hex

    def test_decode_camera_brightness_reduction(self):
        """Camera photo of screen reduces white to ~220-230. Decoder must adapt."""
        import numpy as np

        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        glyph_png = render_png(tree, "hexagon", size=256)

        glyph_img = Image.open(io.BytesIO(glyph_png)).convert("RGBA")
        frame = Image.new("RGBA", (512, 512), (15, 23, 42, 255))

        card_w, card_h = 300, 300
        card_x = (512 - card_w) // 2
        card_y = 80
        white_card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
        frame.paste(white_card, (card_x, card_y))

        glyph_x = card_x + (card_w - 256) // 2
        glyph_y = card_y + (card_h - 256) // 2
        frame.alpha_composite(glyph_img, dest=(glyph_x, glyph_y))

        # Simulate camera brightness reduction
        arr = np.array(frame.convert("RGB"), dtype=np.float32)
        arr = arr * 0.88 + 10
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        dimmed = Image.fromarray(arr, "RGB")

        buf = io.BytesIO()
        dimmed.save(buf, format="JPEG", quality=85)

        result = decode_image(buf.getvalue())
        assert result.confidence == 0.95, f"Brightness-reduced failed: {result.error}"
        assert result.data_hex == data_hex


class TestParseGlyphType:
    def test_parse_member_type(self):
        from src.decoder import parse_glyph_type

        pubkey = "0102030405060708090a0b0c"
        tier = "02"
        data_hex = "01" + pubkey + tier
        result = parse_glyph_type(data_hex)
        assert result["type"] == "MEMBER"
        assert result["pubkey_prefix"] == pubkey
        assert result["tier"] == 2
        assert result["raw_hex"] == data_hex

    def test_parse_community_type(self):
        from src.decoder import parse_glyph_type

        community_id = "aabbccdd"
        join_req = "01"
        timestamp = "65538d80"
        presenter = "11223344"
        data_hex = "02" + community_id + join_req + timestamp + presenter
        result = parse_glyph_type(data_hex)
        assert result["type"] == "COMMUNITY"
        assert result["community_id_prefix"] == community_id
        assert result["join_requirement"] == 1
        assert result["expiry_timestamp"] == int(timestamp, 16)
        assert result["presenter_prefix"] == presenter

    def test_parse_legacy_no_type_byte(self):
        from src.decoder import parse_glyph_type

        data_hex = "00aabbccddeeff1122334455"  # pragma: allowlist secret
        result = parse_glyph_type(data_hex)
        assert result["type"] == "LEGACY"

    def test_parse_legacy_for_existing_16_byte_data(self):
        from src.decoder import parse_glyph_type

        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        result = parse_glyph_type(data_hex)
        assert result["type"] == "LEGACY"

    def test_parse_invalid_hex(self):
        from src.decoder import parse_glyph_type

        result = parse_glyph_type("not_valid_hex")
        assert result["type"] == "LEGACY"

    def test_parse_empty_string(self):
        from src.decoder import parse_glyph_type

        result = parse_glyph_type("")
        assert result["type"] == "LEGACY"

    def test_parse_invite_type_recognized(self):
        from src.decoder import parse_glyph_type

        data_hex = "03" + "aa" * 13
        result = parse_glyph_type(data_hex)
        assert result["type"] == "INVITE"

    def test_parse_recovery_type_recognized(self):
        from src.decoder import parse_glyph_type

        data_hex = "04" + "bb" * 13
        result = parse_glyph_type(data_hex)
        assert result["type"] == "RECOVERY"


class TestTypedPayloadRoundtrip:
    """End-to-end: encode typed payload -> render PNG -> decode -> parse type."""

    def test_member_roundtrip(self):
        from src.decoder import decode_image, parse_glyph_type
        from src.encoder import encode, encode_member_payload
        from src.renderer import render_png

        pubkey = bytes(range(1, 13))
        tier = 2
        data_hex = encode_member_payload(pubkey, tier=tier)
        tree = encode(data_hex, "hexagon")
        png_bytes = render_png(tree, "hexagon", size=512)

        result = decode_image(png_bytes)
        assert result.confidence == 0.95, f"Decode failed: {result.error}"
        assert result.data_hex == data_hex

        parsed = parse_glyph_type(result.data_hex)
        assert parsed["type"] == "MEMBER"
        assert parsed["pubkey_prefix"] == pubkey.hex()
        assert parsed["tier"] == tier

    def test_community_roundtrip(self):
        from src.decoder import decode_image, parse_glyph_type
        from src.encoder import encode, encode_community_payload
        from src.renderer import render_png

        community_id = b"\xaa\xbb\xcc\xdd"
        presenter = b"\x11\x22\x33\x44"
        ts = 1700000000
        join_req = 1

        data_hex = encode_community_payload(
            community_id_prefix=community_id,
            join_requirement=join_req,
            expiry_timestamp=ts,
            presenter_prefix=presenter,
        )
        tree = encode(data_hex, "circle")
        png_bytes = render_png(tree, "circle", size=512)

        result = decode_image(png_bytes)
        assert result.confidence == 0.95, f"Decode failed: {result.error}"
        assert result.data_hex == data_hex

        parsed = parse_glyph_type(result.data_hex)
        assert parsed["type"] == "COMMUNITY"
        assert parsed["community_id_prefix"] == community_id.hex()
        assert parsed["join_requirement"] == join_req
        assert parsed["expiry_timestamp"] == ts
        assert parsed["presenter_prefix"] == presenter.hex()

    def test_member_roundtrip_all_shapes(self):
        from src.decoder import decode_image, parse_glyph_type
        from src.encoder import encode, encode_member_payload
        from src.renderer import render_png

        pubkey = bytes(range(10, 22))
        data_hex = encode_member_payload(pubkey, tier=1)

        for shape in [
            "hexagon",
            "circle",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            tree = encode(data_hex, shape)
            png_bytes = render_png(tree, shape, size=512)
            result = decode_image(png_bytes)
            assert result.confidence == 0.95, f"Shape {shape} failed: {result.error}"
            assert result.data_hex == data_hex, f"Shape {shape}: wrong data"
            parsed = parse_glyph_type(result.data_hex)
            assert parsed["type"] == "MEMBER", f"Shape {shape}: wrong type"

    def test_member_roundtrip_all_tiers(self):
        from src.decoder import decode_image, parse_glyph_type
        from src.encoder import encode, encode_member_payload
        from src.renderer import render_png

        pubkey = bytes(range(20, 32))
        for tier in range(4):
            data_hex = encode_member_payload(pubkey, tier=tier)
            tree = encode(data_hex, "hexagon")
            png_bytes = render_png(tree, "hexagon", size=512, tier=tier)
            result = decode_image(png_bytes)
            assert result.confidence == 0.95, f"Tier {tier} failed: {result.error}"
            assert result.data_hex == data_hex, f"Tier {tier}: wrong data"
            parsed = parse_glyph_type(result.data_hex)
            assert parsed["tier"] == tier, f"Tier {tier}: parsed wrong tier"

    def test_legacy_payload_still_works(self):
        """Existing untyped payloads must still encode/decode correctly."""
        from src.decoder import decode_image, parse_glyph_type
        from src.encoder import encode
        from src.renderer import render_png

        data_hex = "a3f04b2c9e1d7f8a0011223344556677"  # pragma: allowlist secret
        tree = encode(data_hex, "hexagon")
        png_bytes = render_png(tree, "hexagon", size=512)

        result = decode_image(png_bytes)
        assert result.confidence == 0.95
        assert result.data_hex == data_hex

        parsed = parse_glyph_type(result.data_hex)
        assert parsed["type"] == "LEGACY"
