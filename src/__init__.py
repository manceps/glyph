"""Glyph -- radial-ring topology encoder/decoder for visual codes.

A Python implementation of Glyph visual codes using a radial-ring
layout. Encodes arbitrary binary data as geometric glyph images that
are robust to camera capture, JPEG compression, and varying lighting.

Based on the Claycode concept by Marco Maida et al. (ACM TOG / SIGGRAPH 2025).
This implementation uses a radial-ring cell layout instead of the original
nested-tree approach, optimized for reliable camera decoding.

See: https://github.com/marcomaida/claycode
"""
