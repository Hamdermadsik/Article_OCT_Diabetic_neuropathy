"""Index raw DICOM acquisitions by reading headers only.

Reading a full 194 MB volume costs seconds; reading its header costs
milliseconds. Every field the manifest needs -- geometry, transfer syntax, and
the byte offset where pixel data begins -- comes from the header, so the whole
drive can be indexed without decoding a single frame.

The recorded ``pixel_data_offset`` lets later stages seek directly to an
individual frame instead of decoding the volume around it.
"""

from __future__ import annotations

from pathlib import Path

import pydicom

from .cohort_specs import (
    EXPECTED_BITS_ALLOCATED,
    EXPECTED_COLUMNS,
    EXPECTED_FRAMES,
    EXPECTED_ROWS,
    EXPECTED_SAMPLES_PER_PIXEL,
    UNCOMPRESSED_TRANSFER_SYNTAXES,
)

PIXEL_DATA_TAG_LE = b"\xe0\x7f\x10\x00"
_EXPLICIT_VR_CODES = {b"OB", b"OW", b"OF", b"UN"}
_OFFSET_SEARCH_LIMIT = 8 * 1024 * 1024


def is_junk_filename(name: str) -> bool:
    """macOS AppleDouble sidecars and Windows zone-identifier streams.

    Both masquerade as real files in directory listings; counting them is what
    made earlier tallies of this dataset wrong.
    """
    return name.startswith("._") or "Zone.Identifier" in name


def iter_dicom_files(root: Path):
    """Yield real ``.dcm`` files under ``root``, skipping sidecar junk."""
    for path in sorted(root.rglob("*.dcm")):
        if not path.is_file() or is_junk_filename(path.name):
            continue
        yield path


def find_pixel_data_offset(path: Path, limit: int = _OFFSET_SEARCH_LIMIT) -> int | None:
    """Byte offset of the first pixel byte, or ``None`` if not locatable.

    Scans for the PixelData tag and steps over whichever length encoding the
    transfer syntax uses (implicit VR: 4-byte length; explicit VR with a binary
    VR: 2-byte VR + 2 reserved + 4-byte length).
    """
    # The header is normally ~1.5 KB, so start small and only read further when
    # the tag has not appeared yet. Reading a fixed multi-megabyte block per
    # file turns a whole-drive index into tens of gigabytes of I/O.
    buffer = b""
    chunk = 64 * 1024
    index = -1
    with path.open("rb") as handle:
        while len(buffer) < limit:
            data = handle.read(min(chunk, limit - len(buffer)))
            if not data:
                break
            buffer += data
            index = buffer.find(PIXEL_DATA_TAG_LE)
            # Need the tag plus up to 8 bytes after it to decode the length field.
            if index != -1 and len(buffer) >= index + len(PIXEL_DATA_TAG_LE) + 8:
                break
            chunk *= 2

    if index == -1:
        return None

    after_tag = index + len(PIXEL_DATA_TAG_LE)
    if buffer[after_tag : after_tag + 2] in _EXPLICIT_VR_CODES:
        return after_tag + 8  # VR (2) + reserved (2) + length (4)
    return after_tag + 4  # implicit VR: 4-byte length


def frame_byte_size(rows: int, columns: int, samples_per_pixel: int, bits_allocated: int) -> int:
    """Bytes occupied by a single frame's pixel data."""
    return rows * columns * samples_per_pixel * (bits_allocated // 8)


def read_dicom_header(path: Path) -> dict:
    """Read one acquisition's header and validate it against expectations.

    Never raises for a bad file: problems are reported via ``header_status`` so
    a single corrupt acquisition cannot abort a whole-drive index.
    """
    record = {
        "filepath": str(path),
        "filename": path.name,
        "file_size": None,
        "num_frames": None,
        "rows": None,
        "cols": None,
        "samples_per_pixel": None,
        "bits_allocated": None,
        "transfer_syntax": None,
        "pixel_data_offset": None,
        "matches_expected_geometry": False,
        "header_status": "ok",
        "header_detail": "",
    }

    try:
        record["file_size"] = path.stat().st_size
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception as exc:  # noqa: BLE001 - any read failure is a data problem
        record["header_status"] = "unreadable"
        record["header_detail"] = f"{type(exc).__name__}: {exc}"
        return record

    try:
        record["num_frames"] = int(getattr(dataset, "NumberOfFrames", 1) or 1)
        record["rows"] = int(dataset.Rows)
        record["cols"] = int(dataset.Columns)
        record["samples_per_pixel"] = int(dataset.SamplesPerPixel)
        record["bits_allocated"] = int(dataset.BitsAllocated)
        record["transfer_syntax"] = str(dataset.file_meta.TransferSyntaxUID)
    except Exception as exc:  # noqa: BLE001
        record["header_status"] = "missing_geometry"
        record["header_detail"] = f"{type(exc).__name__}: {exc}"
        return record

    # Geometry is recorded as observed rather than checked against a single
    # blessed shape: the two cohorts genuinely differ (May volumes are
    # 460x1174, at least some October volumes are 460x1129). Variation is a
    # fact the pipeline must handle, not an error. ``header_status`` is
    # reserved for problems that actually block using a file.
    record["matches_expected_geometry"] = (
        record["rows"] == EXPECTED_ROWS
        and record["cols"] == EXPECTED_COLUMNS
        and record["num_frames"] == EXPECTED_FRAMES
        and record["samples_per_pixel"] == EXPECTED_SAMPLES_PER_PIXEL
        and record["bits_allocated"] == EXPECTED_BITS_ALLOCATED
    )

    if record["transfer_syntax"] not in UNCOMPRESSED_TRANSFER_SYNTAXES:
        record["header_status"] = "compressed"
        record["header_detail"] = f"transfer syntax {record['transfer_syntax']}"
        return record

    offset = find_pixel_data_offset(path)
    record["pixel_data_offset"] = offset
    if offset is None:
        record["header_status"] = "offset_not_found"
        record["header_detail"] = "PixelData tag not located within search limit"
        return record

    expected_bytes = record["num_frames"] * frame_byte_size(
        record["rows"], record["cols"], record["samples_per_pixel"], record["bits_allocated"]
    )
    if offset + expected_bytes > record["file_size"]:
        record["header_status"] = "truncated"
        record["header_detail"] = (
            f"needs {offset + expected_bytes} bytes, file has {record['file_size']}"
        )
    return record
