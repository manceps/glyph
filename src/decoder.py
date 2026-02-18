"""Image decoder for Glyph visual codes.

Decodes a Glyph image back to the original hex data by:
1. Compositing onto white background (transparent regions -> white)
2. Detecting the glyph bounding box and cropping to it
3. For each candidate crop + data_length:
   a. Recompute the radial ring layout to get cell positions
   b. Sample image intensity at each cell centroid
   c. Compute adaptive threshold on cell intensities
   d. Classify bits and validate CRC-32
4. Return the first combination that passes CRC

For camera images (phone scanning a screen), the decoder tries multiple
crop sizes around the detected glyph center, since the exact canvas
boundaries are hard to determine from noisy camera captures.

Cell positions are fully deterministic (independent of data content),
so this is simply: detect glyph -> crop -> recompute positions -> sample -> validate CRC.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import structlog
from PIL import Image, ImageFilter

from .bits import (
    GLYPH_TYPE_COMMUNITY,
    GLYPH_TYPE_INVITE,
    GLYPH_TYPE_MEMBER,
    GLYPH_TYPE_NAMES,
    GLYPH_TYPE_RECOVERY,
    bits_to_hex,
    validate_crc,
)
from .encoder import CELL_RADIUS_FRACTION, CENTER, INNER_RADIUS, OUTER_RADIUS, compute_ring_layout

logger = structlog.get_logger(__name__)


@dataclass
class DecodeResult:
    """Result of decoding a Glyph image.

    Attributes:
        data_hex: Decoded hex data (without CRC), or None if decode failed.
        confidence: Confidence score in [0.0, 1.0].
        error: Error message if decode failed.
    """

    data_hex: str | None
    confidence: float
    error: str | None = None


@dataclass
class _CameraInfo:
    """Detection info for camera images, enabling multi-crop decode."""

    composited: Image.Image
    gray_full: np.ndarray  # grayscale of composited, for direct sampling
    cx: int
    cy: int
    canvas_estimate: int
    img_w: int
    img_h: int
    # Secondary center candidates (card center, etc.)
    alt_centers: list[tuple[int, int]] = field(default_factory=list)


def _detect_and_crop(
    img: Image.Image,
) -> tuple[Image.Image | None, _CameraInfo | None]:
    """Detect the glyph region and crop the image to a tight square.

    Returns:
        Tuple of (cropped 512x512 image or None, camera info or None).
        Camera info is provided for camera-like images so the caller
        can attempt multiple crop sizes if the initial crop fails.
    """
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_bg, img)
    gray = np.array(composited.convert("L"))
    h, w = gray.shape

    # --- Early rejection: mostly-dark images contain no glyph ---
    dark_pixel_count = int(np.sum(gray < 100))
    if dark_pixel_count > 0.80 * gray.size:
        light_pixel_count = int(np.sum(gray > 200))
        if light_pixel_count < 0.05 * gray.size:
            logger.warning(
                "detect_rejected_dark_image",
                dark_pct=round(dark_pixel_count / gray.size * 100, 1),
                light_pct=round(light_pixel_count / gray.size * 100, 1),
            )
            return None, None

    # Check if the image has a dark background (camera photo of glyph).
    corner_size = max(5, min(h, w) // 20)
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    corner_median = float(np.median(np.concatenate([c.ravel() for c in corners])))
    has_dark_bg = corner_median < 100

    # Also detect "mixed" camera frames where screen glow raises dark pixels
    # above 100 but there's still a clear card (bright region) against a
    # darker background.
    p25 = int(np.percentile(gray, 25))
    p75 = int(np.percentile(gray, 75))
    has_card_like = (p75 - p25 > 60) and (p75 > 160)

    logger.info(
        "detect_stats",
        corner_median=round(corner_median, 1),
        p25=p25,
        p75=p75,
        has_dark_bg=has_dark_bg,
        has_card_like=has_card_like,
    )

    if has_dark_bg or has_card_like:
        # Dark/mixed background: find the white card region.
        white_mask = gray > 200
        if not np.any(white_mask):
            logger.warning("detect_dark_bg_no_white_card")
            return None, None

        white_rows = np.any(white_mask, axis=1)
        white_cols = np.any(white_mask, axis=0)
        raw_row_min = int(np.where(white_rows)[0][0])
        raw_row_max = int(np.where(white_rows)[0][-1])
        raw_col_min = int(np.where(white_cols)[0][0])
        raw_col_max = int(np.where(white_cols)[0][-1])

        # Tighten card bbox using density profiling.
        # The raw bbox may include scattered white UI elements (nav bar,
        # status bar) outside the actual card. The card itself is a dense
        # contiguous white region, while scattered elements have low density.
        col_white_count = np.sum(white_mask[raw_row_min : raw_row_max + 1, :], axis=0)
        row_white_count = np.sum(white_mask[:, raw_col_min : raw_col_max + 1], axis=1)

        col_peak = int(np.max(col_white_count)) if len(col_white_count) > 0 else 0
        row_peak = int(np.max(row_white_count)) if len(row_white_count) > 0 else 0

        # Threshold at 30% of peak: card columns have many white pixels,
        # scattered UI elements have far fewer.
        col_thresh = max(col_peak * 0.30, 5)
        row_thresh = max(row_peak * 0.30, 5)

        dense_cols = col_white_count >= col_thresh
        dense_rows = row_white_count >= row_thresh

        if np.any(dense_cols) and np.any(dense_rows):
            card_col_min = int(np.where(dense_cols)[0][0])
            card_col_max = int(np.where(dense_cols)[0][-1])
            card_row_min = int(np.where(dense_rows)[0][0])
            card_row_max = int(np.where(dense_rows)[0][-1])
        else:
            card_col_min, card_col_max = raw_col_min, raw_col_max
            card_row_min, card_row_max = raw_row_min, raw_row_max

        card_h_px = card_row_max - card_row_min
        card_w_px = card_col_max - card_col_min
        if card_h_px < h * 0.1 or card_w_px < w * 0.1:
            logger.warning("detect_dark_bg_card_too_small", card_w=card_w_px, card_h=card_h_px)
            return None, None

        card_cx = (card_col_min + card_col_max) // 2
        card_cy = (card_row_min + card_row_max) // 2

        # Find glyph within card using brightness-band masking.
        # Text labels are very dark (gray < 40), card background is very
        # bright (> white - 50), but glyph elements (border, dots, outline)
        # fall in a distinctive middle band.
        card_region = gray[card_row_min : card_row_max + 1, card_col_min : card_col_max + 1]
        card_pil = Image.fromarray(card_region)
        blurred = np.array(card_pil.filter(ImageFilter.GaussianBlur(radius=2)))

        card_bright = card_region[card_region > 200]
        actual_white = int(np.median(card_bright)) if len(card_bright) > 0 else 255
        upper_thresh = actual_white - 50

        glyph_mask = (blurred > 40) & (blurred < upper_thresh)

        if np.any(glyph_mask):
            rows_idx, cols_idx = np.where(glyph_mask)

            # Percentile bbox (2-98%) to exclude scattered outlier pixels
            g_row_min = int(np.percentile(rows_idx, 2))
            g_row_max = int(np.percentile(rows_idx, 98))
            g_col_min = int(np.percentile(cols_idx, 2))
            g_col_max = int(np.percentile(cols_idx, 98))

            glyph_w = g_col_max - g_col_min
            glyph_h = g_row_max - g_row_min

            glyph_cx = (g_col_min + g_col_max) // 2
            glyph_cy = (g_row_min + g_row_max) // 2

            content_cx = card_col_min + glyph_cx
            content_cy = card_row_min + glyph_cy

            canvas_estimate = min(glyph_w, glyph_h) + 4
            canvas_estimate = min(canvas_estimate, min(card_w_px, card_h_px))

            logger.debug(
                "glyph_band_detection",
                glyph_cx=content_cx,
                glyph_cy=content_cy,
                card_cx=card_cx,
                card_cy=card_cy,
                glyph_w=glyph_w,
                glyph_h=glyph_h,
                canvas_estimate=canvas_estimate,
            )
        else:
            content_cx = card_cx
            content_cy = card_cy
            canvas_estimate = min(card_w_px, card_h_px)

        # Primary center from glyph detection, card center as fallback.
        alt_centers = []
        if (content_cx, content_cy) != (card_cx, card_cy):
            alt_centers.append((card_cx, card_cy))
        camera_info = _CameraInfo(
            composited, gray, content_cx, content_cy, canvas_estimate, w, h, alt_centers
        )
        cropped = _make_crop(composited, content_cx, content_cy, canvas_estimate, w, h)

        logger.debug(
            "glyph_detected_and_cropped",
            original=f"{w}x{h}",
            cx=content_cx,
            cy=content_cy,
            card_cx=card_cx,
            card_cy=card_cy,
            canvas=canvas_estimate,
        )

        return cropped, camera_info

    # Light background path: find non-white pixels (glyph strokes and border)
    mask = gray < 240
    if not np.any(mask):
        return composited, None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # The SVG border is always at the canvas edge, so span ~ canvas size.
    span = max(int(row_max - row_min), int(col_max - col_min))
    canvas_est = span + 4
    canvas_est = min(canvas_est, min(h, w))

    cx = (int(col_min) + int(col_max)) // 2
    cy = (int(row_min) + int(row_max)) // 2
    cropped = _make_crop(composited, cx, cy, canvas_est, w, h)

    logger.debug(
        "glyph_detected_and_cropped",
        original=f"{w}x{h}",
        cx=cx,
        cy=cy,
        canvas=canvas_est,
    )

    return cropped, None


def _make_crop(
    img: Image.Image,
    cx: int,
    cy: int,
    crop_size: int,
    img_w: int,
    img_h: int,
) -> Image.Image:
    """Crop a square region centered at (cx, cy) and resize to 512x512."""
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half + 1)
    y2 = min(img_h, cy + half + 1)

    cropped = img.crop((x1, y1, x2, y2))
    return cropped.resize((512, 512), Image.Resampling.LANCZOS)


def _sample_radial_intensities(
    gray: np.ndarray,
    size: int,
    total_bits: int,
) -> list[int]:
    """Sample raw image intensity at radial cell centroid positions.

    Same ring layout as the encoder.  Returns raw intensity values
    so the caller can apply adaptive thresholding.
    """
    rings = compute_ring_layout(total_bits)

    ring_count = len(rings)
    ring_spacing = (OUTER_RADIUS - INNER_RADIUS) / max(ring_count, 1)
    cell_r = ring_spacing * CELL_RADIUS_FRACTION
    sample_r = max(2, int(cell_r * 0.4 * size))

    intensities: list[int] = []
    h, w = gray.shape

    for _ring_idx, (cells_in_ring, ring_radius) in enumerate(rings):
        for cell_idx in range(cells_in_ring):
            if len(intensities) >= total_bits:
                break

            angle = (2 * math.pi * cell_idx / cells_in_ring) - (math.pi / 2)
            cell_cx = CENTER + ring_radius * math.cos(angle)
            cell_cy = CENTER + ring_radius * math.sin(angle)

            px = int(cell_cx * size)
            py = int(cell_cy * size)

            y_lo = max(0, py - sample_r)
            y_hi = min(h, py + sample_r + 1)
            x_lo = max(0, px - sample_r)
            x_hi = min(w, px + sample_r + 1)

            if y_lo < y_hi and x_lo < x_hi:
                region = gray[y_lo:y_hi, x_lo:x_hi]
                intensities.append(int(np.mean(region)))
            else:
                intensities.append(255)

        if len(intensities) >= total_bits:
            break

    return intensities


def _adaptive_threshold(values: list[int]) -> int:
    """Compute adaptive threshold for separating filled from empty cells.

    - Clean images (<=12 unique values): Walk from bright end, find first gap >= 10.
    - Camera images (>12 unique values): Otsu + gap midpoint.
    """
    if len(values) < 4:
        return 200

    sorted_unique = sorted(set(values))

    if len(sorted_unique) < 2:
        return sorted_unique[0]

    if sorted_unique[-1] - sorted_unique[0] < 15:
        return (sorted_unique[0] + sorted_unique[-1]) // 2

    if len(sorted_unique) <= 12:
        for i in range(len(sorted_unique) - 1, 0, -1):
            gap = sorted_unique[i] - sorted_unique[i - 1]
            if gap >= 10:
                return (sorted_unique[i - 1] + sorted_unique[i]) // 2
        return (sorted_unique[-2] + sorted_unique[-1]) // 2

    # Camera images: Otsu + gap midpoint
    arr = np.array(values, dtype=np.uint8)
    lo, hi = int(arr.min()), int(arr.max())

    hist = np.bincount(arr, minlength=256).astype(np.float64)
    total = len(arr)
    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))

    otsu_t = (lo + hi) // 2
    max_var = 0.0
    sum_bg = 0.0
    weight_bg = 0

    for t in range(lo, hi + 1):
        weight_bg += int(hist[t])
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * float(hist[t])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var = float(weight_bg) * float(weight_fg) * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var = var
            otsu_t = t

    lower = [v for v in values if v <= otsu_t]
    upper = [v for v in values if v > otsu_t]

    if lower and upper:
        return (max(lower) + min(upper)) // 2

    return otsu_t


def _try_decode_on_gray(
    gray: np.ndarray,
    data_bytes_list: list[int],
) -> tuple[str, int, int] | None:
    """Try to decode from a 512x512 grayscale image.

    Returns (data_hex, data_bytes, threshold) on CRC match, or None.
    """
    size = gray.shape[0]

    for data_bytes in data_bytes_list:
        total_bits = (data_bytes + 4) * 8  # +4 for CRC-32
        intensities = _sample_radial_intensities(gray, size, total_bits)
        if len(intensities) < 40:  # minimum: 1 data byte + 4 CRC bytes = 5 bytes = 40 bits
            continue

        # Reject low-contrast regions (uniform color or noise)
        if max(intensities) - min(intensities) < 20:
            continue

        threshold = _adaptive_threshold(intensities)
        bits = [1 if v < threshold else 0 for v in intensities]

        hex_with_crc = bits_to_hex(bits)
        if validate_crc(hex_with_crc):
            data_hex = hex_with_crc[:-8]  # strip 4-byte CRC-32
            return data_hex, data_bytes, threshold

    return None


def _try_decode_direct(
    gray_full: np.ndarray,
    cx: int,
    cy: int,
    crop_size: int,
    data_bytes_list: list[int],
) -> tuple[str, int, int] | None:
    """Try to decode by sampling directly from the full image.

    Instead of cropping and resizing to 512x512, compute cell positions
    at the native resolution and sample directly. Much faster (~5x)
    since it avoids PIL crop+resize per candidate.
    """
    h_full, w_full = gray_full.shape
    half = crop_size // 2

    for data_bytes in data_bytes_list:
        total_bits = (data_bytes + 4) * 8
        rings = compute_ring_layout(total_bits)

        ring_count = len(rings)
        ring_spacing = (OUTER_RADIUS - INNER_RADIUS) / max(ring_count, 1)
        cell_r = ring_spacing * CELL_RADIUS_FRACTION
        sample_r = max(1, int(cell_r * 0.4 * crop_size))

        intensities: list[int] = []

        for cells_in_ring, ring_radius in rings:
            for cell_idx in range(cells_in_ring):
                if len(intensities) >= total_bits:
                    break

                angle = (2 * math.pi * cell_idx / cells_in_ring) - (math.pi / 2)
                cell_cx_norm = CENTER + ring_radius * math.cos(angle)
                cell_cy_norm = CENTER + ring_radius * math.sin(angle)

                # Map normalized position to full-image coordinates
                px = cx - half + int(cell_cx_norm * crop_size)
                py = cy - half + int(cell_cy_norm * crop_size)

                y_lo = max(0, py - sample_r)
                y_hi = min(h_full, py + sample_r + 1)
                x_lo = max(0, px - sample_r)
                x_hi = min(w_full, px + sample_r + 1)

                if y_lo < y_hi and x_lo < x_hi:
                    region = gray_full[y_lo:y_hi, x_lo:x_hi]
                    intensities.append(int(np.mean(region)))
                else:
                    intensities.append(255)

            if len(intensities) >= total_bits:
                break

        if len(intensities) < 40:
            continue

        if max(intensities) - min(intensities) < 20:
            continue

        threshold = _adaptive_threshold(intensities)
        bits = [1 if v < threshold else 0 for v in intensities]

        hex_with_crc = bits_to_hex(bits)
        if validate_crc(hex_with_crc):
            data_hex = hex_with_crc[:-8]
            return data_hex, data_bytes, threshold

    return None


def decode_image(image_bytes: bytes) -> DecodeResult:
    """Decode a Glyph from an image.

    Supports PNG, JPEG, and WebP input. Handles both clean digital
    images and noisy camera captures (phone scanning a screen).

    For clean digital images, tries a single crop with all data lengths.
    For camera images, tries multiple crop sizes around the detected
    glyph center to compensate for imprecise boundary detection.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, or WebP).

    Returns:
        DecodeResult with data_hex, confidence, and optional error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        logger.warning("decode_image_open_failed", error=str(e))
        return DecodeResult(data_hex=None, confidence=0.0, error=f"Cannot open image: {e}")

    # Detect glyph region
    cropped, camera_info = _detect_and_crop(img)
    if cropped is None:
        return DecodeResult(
            data_hex=None,
            confidence=0.0,
            error="No glyph detected -- image is too dark or contains no glyph",
        )

    gray = np.array(cropped.convert("L"))

    # Basic sanity checks on the cropped image
    non_white = gray[gray < 250]
    if len(non_white) == 0:
        return DecodeResult(data_hex=None, confidence=0.0, error="Image appears blank")

    dark_pixels = gray[gray < 100]
    if len(dark_pixels) > 0.70 * gray.size:
        return DecodeResult(
            data_hex=None,
            confidence=0.0,
            error="Image appears too dark -- no glyph content found",
        )

    # Common payload sizes to try (application-specific sizes can be added)
    priority_sizes = [14, 16, 32]

    # --- Attempt 1: single crop (works for clean digital images) ---
    result = _try_decode_on_gray(gray, priority_sizes)
    if result:
        data_hex, data_bytes, threshold = result
        logger.info("decode_success", data_bytes=data_bytes, threshold=threshold, crop="initial")
        return DecodeResult(data_hex=data_hex, confidence=0.95)

    # --- Attempt 2: Grid search using direct sampling (no crop+resize) ---
    if camera_info:
        ci = camera_info

        # Scales: fine near 1.0, extending down to 0.64 for zoom variation.
        crop_scales = [
            1.00,
            0.97,
            1.03,
            0.94,
            1.06,
            0.91,
            1.09,
            0.88,
            1.12,
            0.85,
            1.15,
            0.82,
            0.79,
            0.76,
            0.73,
            0.70,
            0.67,
            0.64,
        ]

        candidate_centers = [(ci.cx, ci.cy)]
        candidate_centers.extend(ci.alt_centers)

        logger.debug(
            "grid_search_start",
            canvas_estimate=ci.canvas_estimate,
            centers=len(candidate_centers),
            primary_center=(ci.cx, ci.cy),
        )

        # Phase 1: +/-30px in 10px steps (49 offsets x 18 scales x N centers)
        # Uses direct sampling (no PIL) -- very fast.
        tight_1d = list(range(-30, 31, 10))
        tight_offsets = [(dx, dy) for dx in tight_1d for dy in tight_1d]

        for base_cx, base_cy in candidate_centers:
            for scale in crop_scales:
                crop_size = max(50, int(ci.canvas_estimate * scale))
                for dx, dy in tight_offsets:
                    result = _try_decode_direct(
                        ci.gray_full,
                        base_cx + dx,
                        base_cy + dy,
                        crop_size,
                        priority_sizes,
                    )
                    if result:
                        data_hex, data_bytes, threshold = result
                        logger.info(
                            "decode_success",
                            data_bytes=data_bytes,
                            threshold=threshold,
                            crop=f"tight({base_cx},{base_cy})_s{scale:.2f}_d({dx},{dy})",
                            crop_size=crop_size,
                        )
                        return DecodeResult(data_hex=data_hex, confidence=0.95)

        # Phase 2: +/-90px in 15px steps (excluding phase 1 area)
        # Wider coverage for frames where detection is off.
        wide_1d = list(range(-90, 91, 15))
        wide_offsets = [
            (dx, dy) for dx in wide_1d for dy in wide_1d if abs(dx) > 30 or abs(dy) > 30
        ]

        for base_cx, base_cy in candidate_centers:
            for scale in crop_scales:
                crop_size = max(50, int(ci.canvas_estimate * scale))
                for dx, dy in wide_offsets:
                    result = _try_decode_direct(
                        ci.gray_full,
                        base_cx + dx,
                        base_cy + dy,
                        crop_size,
                        priority_sizes,
                    )
                    if result:
                        data_hex, data_bytes, threshold = result
                        logger.info(
                            "decode_success",
                            data_bytes=data_bytes,
                            threshold=threshold,
                            crop=f"wide({base_cx},{base_cy})_s{scale:.2f}_d({dx},{dy})",
                            crop_size=crop_size,
                        )
                        return DecodeResult(data_hex=data_hex, confidence=0.95)

    # No CRC-32 match
    logger.warning("decode_no_crc_match", camera=camera_info is not None)
    return DecodeResult(
        data_hex=None,
        confidence=0.0,
        error="No valid glyph found -- CRC check failed on all attempts",
    )


def parse_glyph_type(data_hex: str) -> dict:
    """Parse decoded glyph data to determine type and extract fields.

    After decoding raw hex data from a glyph image, this function inspects
    the first byte to determine the glyph type and extracts type-specific
    fields.

    For backward compatibility, if the first byte does not match a known
    glyph type, the payload is treated as a legacy (untyped) payload.

    Args:
        data_hex: Decoded hex string (without CRC).

    Returns:
        Dict with at minimum:
        - 'type': 'MEMBER' | 'COMMUNITY' | 'INVITE' | 'RECOVERY' | 'LEGACY'
        - 'raw_hex': the original hex string

        For MEMBER type (14 bytes):
        - 'pubkey_prefix': hex string of 12-byte pubkey hash prefix
        - 'tier': int (0-3)

        For COMMUNITY type (14 bytes):
        - 'community_id_prefix': hex string of 4-byte community ID prefix
        - 'join_requirement': int (0-3)
        - 'expiry_timestamp': int (unix timestamp)
        - 'presenter_prefix': hex string of 4-byte presenter pubkey prefix
    """
    result: dict = {"raw_hex": data_hex}

    try:
        data = bytes.fromhex(data_hex)
    except ValueError:
        result["type"] = "LEGACY"
        return result

    if len(data) < 1:
        result["type"] = "LEGACY"
        return result

    type_byte = data[0]
    type_name = GLYPH_TYPE_NAMES.get(type_byte)

    if type_name is None:
        result["type"] = "LEGACY"
        return result

    result["type"] = type_name

    if type_byte == GLYPH_TYPE_MEMBER and len(data) == 14:
        result["pubkey_prefix"] = data[1:13].hex()
        result["tier"] = data[13]
    elif type_byte == GLYPH_TYPE_COMMUNITY and len(data) == 14:
        result["community_id_prefix"] = data[1:5].hex()
        result["join_requirement"] = data[5]
        result["expiry_timestamp"] = int.from_bytes(data[6:10], "big")
        result["presenter_prefix"] = data[10:14].hex()
    elif type_byte == GLYPH_TYPE_INVITE:
        if len(data) >= 17:
            result["community_id_prefix"] = data[1:5].hex()
            result["invite_nonce"] = data[5:13].hex()
            result["inviter_prefix"] = data[13:17].hex()
    elif type_byte == GLYPH_TYPE_RECOVERY:
        if len(data) >= 17:
            result["pubkey_hash_prefix"] = data[1:13].hex()
            result["recovery_nonce"] = int.from_bytes(data[13:17], "big")

    return result
