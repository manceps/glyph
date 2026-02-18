# Contributing to Glyph

Thank you for your interest in contributing! This Python implementation brings
Glyph visual codes to server-side environments with a radial-ring encoding
approach optimized for camera-robust decoding.

## About This Port

This is a Python reimplementation of the [Claycode](https://github.com/marcomaida/claycode) concept by Marco Maida et al. (ACM TOG / SIGGRAPH 2025). It uses a different encoding topology (radial rings vs. nested polygons) but shares the core idea of encoding data in stylizable geometric patterns.

## Development Setup

### Prerequisites

- Python 3.11+
- libcairo2 development headers
  - Ubuntu/Debian: `sudo apt-get install libcairo2-dev pkg-config`
  - macOS: `brew install cairo pkg-config`

### Install

```bash
# Clone and install in development mode
git clone https://github.com/manceps/glyph.git
cd glyph
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_decoder.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Run the Service

```bash
uvicorn src.main:app --port 8090 --reload
```

## Project Structure

```
src/
  __init__.py    # Package metadata
  bits.py        # Bit/hex conversion, CRC-32
  tree.py        # TreeNode, TopologyTree data structures
  shapes.py      # 8 boundary shape templates
  encoder.py     # Radial-ring encoder + typed payload helpers
  decoder.py     # Image decoder (camera-robust)
  renderer.py    # SVG/PNG renderer with tier styling
  main.py        # FastAPI application

tests/
  test_bits.py      # Bit utilities and CRC
  test_shapes.py    # Shape templates
  test_encoder.py   # Encoding + typed payloads
  test_decoder.py   # Decoding + camera robustness
  test_renderer.py  # SVG/PNG rendering + tiers
  test_api.py       # HTTP endpoint integration tests
```

## Key Design Decisions

### Why Radial Rings?

The original Claycode uses nested polygon containment (tree topology) to encode data.
Our implementation uses concentric rings of cells because:

1. **Deterministic positions**: Cell coordinates depend only on the total bit count,
   not the data. The decoder knows where to sample without detecting contours.
2. **No contour detection**: Instead of OpenCV contour finding, we sample image
   intensity at known cell positions and apply adaptive thresholding.
3. **Camera robustness**: A grid search over crop offsets (±90px) and scale factors
   (0.64-1.15) compensates for imprecise glyph detection in camera photos.
4. **Direct sampling**: For camera images, we sample directly from the full-resolution
   grayscale array instead of cropping and resizing per candidate, ~100-300x faster.

### CRC-32 vs CRC-16

We use CRC-32 (4 bytes) instead of CRC-16 (2 bytes) for stronger error detection.
With 14-byte payloads, the overhead is acceptable (18 bytes total = 144 bits).

### Adaptive Thresholding

The decoder uses two strategies depending on image quality:
- **Clean images** (≤12 unique intensity values): Walk from bright end, find first gap ≥10
- **Camera images** (>12 unique values): Otsu's method + gap midpoint

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Make your changes with tests
4. Run the test suite (`pytest tests/ -v`)
5. Run the linter (`ruff check src/ tests/`)
6. Submit a pull request with a clear description

## Areas for Contribution

- **Additional shape templates**: New boundary shapes beyond the current 8
- **Color palette generation**: Algorithmic palette derivation from seed data
- **Performance optimization**: Faster decoding for real-time camera scanning
- **WebAssembly build**: Browser-side decoding via Pyodide or similar
- **Additional error correction**: Reed-Solomon or similar forward error correction
- **Documentation**: Usage examples, tutorials, integration guides
