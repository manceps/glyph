"""Glyph microservice -- FastAPI application.

Endpoints:
    POST /encode        -- Encode hex data to PNG image
    POST /encode/svg    -- Encode hex data to SVG string
    POST /encode/typed  -- Encode a typed glyph (MEMBER, COMMUNITY, etc.)
    POST /decode        -- Decode image to hex data
    GET  /health        -- Health check

This service is fully self-contained with no external dependencies
beyond its own Python packages.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .bits import GLYPH_TYPE_BY_NAME
from .decoder import decode_image, parse_glyph_type
from .encoder import (
    encode,
    encode_community_payload,
    encode_invite_payload,
    encode_member_payload,
    encode_recovery_payload,
)
from .renderer import render_png, render_svg

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="glyph",
    description="Glyph radial-ring topology encoder/decoder for visual identity glyphs",
    version="0.2.0",
)


# --------------------------------------------------------------------------
# Request / Response models
# --------------------------------------------------------------------------


class EncodeRequest(BaseModel):
    """Request body for /encode and /encode/svg."""

    data_hex: str = Field(
        ...,
        description="Hex-encoded data to encode (e.g. SHA-256 hash prefix)",
        examples=["a3f04b2c9e1d7f8a"],  # pragma: allowlist secret
    )
    shape: str = Field(
        default="hexagon",
        description="Outer shape template name",
        examples=["hexagon", "circle", "shield", "octagon", "diamond"],
    )
    colors: list[str] = Field(
        default_factory=lambda: [
            "#2D3748",
            "#4A5568",
            "#718096",
            "#A0AEC0",
            "#CBD5E0",
        ],
        description="List of hex color strings for rendering",
    )
    size: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Output image size in pixels (square)",
    )


class TypedEncodeRequest(BaseModel):
    """Request body for /encode/typed."""

    glyph_type: str = Field(
        ...,
        description="Glyph type: MEMBER, COMMUNITY, INVITE, RECOVERY",
        examples=["MEMBER", "COMMUNITY"],
    )
    # MEMBER fields
    pubkey_hash_prefix: str | None = Field(
        default=None,
        description="Hex string of 12-byte pubkey hash prefix (MEMBER type)",
    )
    tier: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Trust tier: 0-3",
    )
    # COMMUNITY fields
    community_id_prefix: str | None = Field(
        default=None,
        description="Hex string of 4-byte community ID prefix (COMMUNITY type)",
    )
    join_requirement: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Join requirement: 0-3",
    )
    expiry_timestamp: int = Field(
        default=0,
        ge=0,
        description="Unix timestamp for glyph expiry (COMMUNITY type)",
    )
    presenter_prefix: str | None = Field(
        default=None,
        description="Hex string of 4-byte presenter pubkey prefix (COMMUNITY type)",
    )
    # RECOVERY fields
    recovery_nonce: int = Field(
        default=0,
        ge=0,
        description="Recovery request nonce (RECOVERY type)",
    )
    # INVITE fields
    invite_nonce: str | None = Field(
        default=None,
        description="Hex string of 8-byte invite nonce (INVITE type)",
    )
    inviter_prefix: str | None = Field(
        default=None,
        description="Hex string of 4-byte inviter pubkey prefix (INVITE type)",
    )
    # Rendering options
    shape: str = Field(
        default="hexagon",
        description="Outer shape template name",
    )
    colors: list[str] = Field(
        default_factory=lambda: [
            "#2D3748",
            "#4A5568",
            "#718096",
            "#A0AEC0",
            "#CBD5E0",
        ],
        description="List of hex color strings for rendering",
    )
    size: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Output image size in pixels (square)",
    )
    format: str = Field(
        default="png",
        description="Output format: png or svg",
    )


class DecodeResponse(BaseModel):
    """Response body for /decode."""

    data_hex: str | None = Field(
        description="Decoded hex data, or null if decode failed",
    )
    confidence: float = Field(
        description="Confidence score in [0.0, 1.0]",
    )
    error: str | None = Field(
        default=None,
        description="Error message if decode failed",
    )
    glyph_type: str | None = Field(
        default=None,
        description="Parsed glyph type: MEMBER, COMMUNITY, INVITE, RECOVERY, LEGACY, or null",
    )
    glyph_fields: dict | None = Field(
        default=None,
        description="Parsed type-specific fields (pubkey_prefix, tier, etc.)",
    )


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str
    service: str
    version: str


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.post(
    "/encode",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG-encoded glyph"},
        422: {"description": "Invalid input"},
    },
)
async def encode_png(request: EncodeRequest) -> Response:
    """Encode hex data into a Glyph PNG image."""
    try:
        tree = encode(request.data_hex, request.shape)
        png_bytes = render_png(tree, request.shape, request.colors, request.size)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("encode_png_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Encoding failed")

    return Response(content=png_bytes, media_type="image/png")


@app.post(
    "/encode/svg",
    response_class=Response,
    responses={
        200: {
            "content": {"image/svg+xml": {}},
            "description": "SVG-encoded glyph",
        },
        422: {"description": "Invalid input"},
    },
)
async def encode_svg_endpoint(request: EncodeRequest) -> Response:
    """Encode hex data into a Glyph SVG image."""
    try:
        tree = encode(request.data_hex, request.shape)
        svg_content = render_svg(tree, request.shape, request.colors, request.size)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("encode_svg_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Encoding failed")

    return Response(content=svg_content, media_type="image/svg+xml")


@app.post(
    "/encode/typed",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {},
                "image/svg+xml": {},
            },
            "description": "Encoded typed glyph",
        },
        422: {"description": "Invalid input"},
    },
)
async def encode_typed(request: TypedEncodeRequest) -> Response:
    """Encode a typed glyph (MEMBER, COMMUNITY, etc.)."""
    glyph_type_upper = request.glyph_type.upper()

    if glyph_type_upper not in GLYPH_TYPE_BY_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown glyph type: {request.glyph_type}. "
            f"Valid types: {', '.join(GLYPH_TYPE_BY_NAME.keys())}",
        )

    try:
        if glyph_type_upper == "MEMBER":
            if not request.pubkey_hash_prefix:
                raise ValueError("pubkey_hash_prefix is required for MEMBER glyph type")
            data_hex = encode_member_payload(
                bytes.fromhex(request.pubkey_hash_prefix),
                request.tier,
            )
            tier = request.tier
            glyph_subtype = None
        elif glyph_type_upper == "COMMUNITY":
            if not request.community_id_prefix:
                raise ValueError("community_id_prefix is required for COMMUNITY glyph type")
            if not request.presenter_prefix:
                raise ValueError("presenter_prefix is required for COMMUNITY glyph type")
            data_hex = encode_community_payload(
                bytes.fromhex(request.community_id_prefix),
                request.join_requirement,
                request.expiry_timestamp,
                bytes.fromhex(request.presenter_prefix),
            )
            tier = request.tier
            glyph_subtype = "community"
        elif glyph_type_upper == "RECOVERY":
            if not request.pubkey_hash_prefix:
                raise ValueError("pubkey_hash_prefix is required for RECOVERY glyph type")
            data_hex = encode_recovery_payload(
                bytes.fromhex(request.pubkey_hash_prefix),
                request.recovery_nonce,
            )
            tier = 0
            glyph_subtype = "recovery"
        elif glyph_type_upper == "INVITE":
            if not request.community_id_prefix:
                raise ValueError("community_id_prefix is required for INVITE glyph type")
            if not request.invite_nonce:
                raise ValueError("invite_nonce is required for INVITE glyph type")
            if not request.inviter_prefix:
                raise ValueError("inviter_prefix is required for INVITE glyph type")
            data_hex = encode_invite_payload(
                bytes.fromhex(request.community_id_prefix),
                bytes.fromhex(request.invite_nonce),
                bytes.fromhex(request.inviter_prefix),
            )
            tier = 0
            glyph_subtype = "invite"
        else:
            raise ValueError(f"Glyph type {glyph_type_upper} encoding is not yet implemented")

        tree = encode(data_hex, request.shape)

        if request.format.lower() == "svg":
            svg_content = render_svg(
                tree,
                request.shape,
                request.colors,
                request.size,
                tier=tier,
                glyph_subtype=glyph_subtype,
            )
            return Response(content=svg_content, media_type="image/svg+xml")
        else:
            png_bytes = render_png(
                tree,
                request.shape,
                request.colors,
                request.size,
                tier=tier,
                glyph_subtype=glyph_subtype,
            )
            return Response(content=png_bytes, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("encode_typed_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Encoding failed")


@app.post("/decode", response_model=DecodeResponse)
async def decode_endpoint(file: UploadFile = File(...)) -> DecodeResponse:
    """Decode a Glyph image back to hex data."""
    if file.content_type and file.content_type not in (
        "image/png",
        "image/jpeg",
        "image/webp",
    ):
        raise HTTPException(
            status_code=422,
            detail=(f"Unsupported image type: {file.content_type}. " "Use PNG, JPEG, or WebP."),
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    result = decode_image(image_bytes)

    # Parse glyph type from decoded data
    glyph_type = None
    glyph_fields = None
    if result.data_hex:
        parsed = parse_glyph_type(result.data_hex)
        glyph_type = parsed.get("type")
        # Extract type-specific fields (everything except 'type' and 'raw_hex')
        glyph_fields = {k: v for k, v in parsed.items() if k not in ("type", "raw_hex")}
        if not glyph_fields:
            glyph_fields = None

    return DecodeResponse(
        data_hex=result.data_hex,
        confidence=result.confidence,
        error=result.error,
        glyph_type=glyph_type,
        glyph_fields=glyph_fields,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for Docker and load balancer probes."""
    return HealthResponse(
        status="healthy",
        service="glyph",
        version="0.2.0",
    )
