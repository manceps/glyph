# glyph

A Python implementation of visual codes using a **radial-ring encoding** layout. Encodes arbitrary binary data (up to 64 bytes) as stylized geometric glyph images that are robust to camera capture, JPEG compression, and varying lighting conditions.

Based on the [Claycode](https://github.com/marcomaida/claycode) concept by Marco Maida et al. ([ACM TOG / SIGGRAPH 2025](https://dl.acm.org/doi/10.1145/3730853)).

## What Are Glyphs?

Glyphs are a novel type of 2D scannable code that encode data in a topology structure rather than a pixel matrix. Unlike QR codes, Glyphs can be heavily stylized, different shapes, colors, and visual treatments, while remaining reliably decodable.

This implementation uses a **radial-ring layout** where data bits are encoded as small polygon cells arranged in concentric rings within a boundary shape:

- **bit=1**: Cell is filled with color (dark)
- **bit=0**: Cell is white (empty)
- **CRC-32**: 4-byte error detection appended to all payloads

The result is an organic, abstract geometric pattern that encodes data while looking visually distinctive and aesthetically pleasing.

## Key Differences from the Original

| Aspect | Original (JS) | This Implementation (Python) |
|--------|---------------|------------------------------|
| **Layout** | Nested polygon tree | Radial concentric rings |
| **Encoding** | Tree branching pattern | Direct bit-per-cell |
| **Error detection** | CRC-16 | CRC-32 |
| **Rendering** | PixiJS (browser) | SVG + CairoSVG (server) |
| **Decoding** | OpenCV.js contours | Intensity sampling + adaptive threshold |
| **Max data** | Variable | 64 bytes |
| **Camera robustness** | Browser camera API | Multi-crop grid search with direct sampling |

The radial-ring approach was chosen because:
1. Cell positions are deterministic (independent of data), enabling reliable decoding by sampling at known coordinates
2. No contour detection needed: just sample intensity at cell centroids
3. Robust to camera noise: adaptive thresholding + grid search over crop offsets and scales
4. Supports JPEG compression artifacts from phone cameras

## Features

- **8 boundary shapes**: circle, hexagon, octagon, shield, diamond, pentagon, heptagon, triangle
- **Custom color palettes**: Full control over glyph colors
- **SVG and PNG output**: Vector and raster rendering
- **Camera-robust decoding**: Handles phone photos of screens with dark backgrounds, brightness variation, and JPEG artifacts
- **Typed payloads**: Extensible type system for structured data (MEMBER, COMMUNITY, INVITE, RECOVERY)
- **Tier-based visuals**: Optional opacity, saturation, and border ring styling
- **Docker-ready**: Self-contained microservice with health check

## Quick Start

### Docker (recommended)

```bash
docker run -p 8090:8090 manceps/glyph:latest
```

Or build from source:

```bash
docker build -t glyph .
docker run -p 8090:8090 glyph
```

### Local Development

```bash
# Requires Python 3.11+ and libcairo2
pip install -e ".[dev]"
uvicorn src.main:app --port 8090
```

### Verify

```bash
curl http://localhost:8090/health
# {"status":"healthy","service":"glyph","version":"0.2.0"}
```

## API Reference

### `POST /encode`: Encode to PNG

```bash
curl -X POST http://localhost:8090/encode \
  -H "Content-Type: application/json" \
  -d '{"data_hex": "deadbeefcafebabe", "shape": "hexagon", "size": 512}' \
  --output glyph.png
```

**Request body:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_hex` | string | (required) | Hex-encoded data (1-64 bytes) |
| `shape` | string | `"hexagon"` | Boundary shape name |
| `colors` | string[] | neutral grays | List of hex color strings |
| `size` | int | `512` | Output size in pixels (64-2048) |

**Response:** `image/png`

### `POST /encode/svg`: Encode to SVG

Same request body as `/encode`. Returns `image/svg+xml`.

### `POST /encode/typed`: Encode Typed Glyph

Encodes structured typed payloads with application-specific fields.

```bash
curl -X POST http://localhost:8090/encode/typed \
  -H "Content-Type: application/json" \
  -d '{
    "glyph_type": "MEMBER",
    "pubkey_hash_prefix": "0102030405060708090a0b0c",
    "tier": 2,
    "shape": "hexagon",
    "size": 512,
    "format": "svg"
  }' --output member.svg
```

**Supported glyph types:**

| Type | Fields | Payload Size |
|------|--------|-------------|
| `MEMBER` | `pubkey_hash_prefix` (12B), `tier` (0-3) | 14 bytes |
| `COMMUNITY` | `community_id_prefix` (4B), `join_requirement` (0-3), `expiry_timestamp`, `presenter_prefix` (4B) | 14 bytes |
| `INVITE` | `community_id_prefix` (4B), `invite_nonce` (8B), `inviter_prefix` (4B) | 17 bytes |
| `RECOVERY` | `pubkey_hash_prefix` (12B), `recovery_nonce` | 17 bytes |

### `POST /decode`: Decode Image

```bash
curl -X POST http://localhost:8090/decode \
  -F "file=@glyph.png" | python -m json.tool
```

**Response:**
```json
{
  "data_hex": "deadbeefcafebabe",
  "confidence": 0.95,
  "error": null,
  "glyph_type": "LEGACY",
  "glyph_fields": null
}
```

For typed glyphs, `glyph_type` will be `"MEMBER"`, `"COMMUNITY"`, etc., and `glyph_fields` will contain the parsed fields.

### `GET /health`: Health Check

```bash
curl http://localhost:8090/health
```

## Python Library Usage

You can also use the encoder/decoder directly as a Python library:

```python
from src.encoder import encode
from src.renderer import render_svg, render_png
from src.decoder import decode_image

# Encode data as SVG
tree = encode("deadbeefcafebabe", "hexagon")
svg = render_svg(tree, "hexagon", colors=["#1A365D", "#2B6CB0", "#4299E1"])

# Encode as PNG
png_bytes = render_png(tree, "hexagon", size=512)

# Decode from image
result = decode_image(png_bytes)
print(result.data_hex)      # "deadbeefcafebabe"
print(result.confidence)     # 0.95
```

### Typed Payloads

```python
from src.encoder import encode, encode_member_payload
from src.decoder import decode_image, parse_glyph_type
from src.renderer import render_png

# Create a typed member glyph
data_hex = encode_member_payload(
    pubkey_hash_prefix=bytes.fromhex("0102030405060708090a0b0c"),
    tier=2,
)
tree = encode(data_hex, "hexagon")
png = render_png(tree, "hexagon", size=512, tier=2)

# Decode and parse type
result = decode_image(png)
parsed = parse_glyph_type(result.data_hex)
print(parsed["type"])           # "MEMBER"
print(parsed["pubkey_prefix"])  # "0102030405060708090a0b0c"
print(parsed["tier"])           # 2
```

## Architecture

```
src/
  bits.py       # Hex/bit conversion, CRC-32, type constants
  tree.py       # TreeNode and TopologyTree data structures
  shapes.py     # 8 boundary shape templates (normalized vertices)
  encoder.py    # Radial-ring layout + typed payload encoders
  decoder.py    # Image detection, intensity sampling, CRC validation
  renderer.py   # SVG/PNG rendering with tier-based styling
  main.py       # FastAPI application (HTTP endpoints)
```

### Encoding Pipeline

```
hex data -> bits -> append CRC-32 -> compute ring layout -> build tree -> render SVG -> (optional) convert to PNG
```

### Decoding Pipeline

```
image -> composite on white -> detect glyph region -> crop to square
      -> sample intensity at cell positions -> adaptive threshold
      -> classify bits -> validate CRC-32 -> extract data
```

For camera images, the decoder performs a grid search over:
- Multiple crop center offsets (tight +/-30px, wide +/-90px)
- 18 scale factors (0.64 to 1.15)
- Direct sampling at native resolution (no PIL crop+resize per candidate)

## Running Tests

```bash
# Unit tests
pytest tests/

# With verbose output
pytest tests/ -v

# Specific test file
pytest tests/test_decoder.py -v
```

## System Dependencies

- **Python 3.11+**
- **libcairo2** (for CairoSVG PNG rendering)
  - Ubuntu/Debian: `apt-get install libcairo2-dev`
  - macOS: `brew install cairo`
  - Alpine: `apk add cairo-dev`

## Extending Glyph Types

The type system is extensible. To add a new glyph type:

1. Reserve a type byte in `bits.py` (e.g., `GLYPH_TYPE_CUSTOM = 0x05`)
2. Add it to `GLYPH_TYPE_NAMES` and `GLYPH_TYPE_BY_NAME`
3. Create an `encode_custom_payload()` function in `encoder.py`
4. Add parsing logic to `parse_glyph_type()` in `decoder.py`
5. Optionally add a visual subtype in `renderer.py`

## Credits

- **Original Claycode concept**: Marco Maida et al., [ACM TOG / SIGGRAPH 2025](https://dl.acm.org/doi/10.1145/3730853)
- **Original implementation**: [github.com/marcomaida/claycode](https://github.com/marcomaida/claycode) (JavaScript/PixiJS/OpenCV.js)
- **Python port**: Radial-ring encoding, camera-robust decoding, typed payloads, FastAPI microservice

## License

MIT License. See [LICENSE](LICENSE) for details.
