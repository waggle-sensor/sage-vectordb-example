#!/usr/bin/env python3
"""
Find rows in any Hugging Face dataset whose image column fails decode/save/encode.

Use this to debug "cannot identify image file", "broken data stream", or similar errors
during data loading. Reads each row without decoding the image first, then tries to
decode; on failure reports the full row. Outputs a CSV of failing rows (index, error,
image_note, plus all other columns except the image bytes).

Usage:
  pip install datasets pillow  # weaviate-client only if --check-weaviate

  # Any dataset: pass --dataset (required) and optional --image-column, --split, --config
  python find_bad_images.py --dataset sagecontinuum/CommonObjectsBench
  python find_bad_images.py --dataset imdb --config plain_text
  python find_bad_images.py --dataset my-org/my-dataset --image-column img --split train

  python find_bad_images.py --dataset <name> --limit 100
  python find_bad_images.py --dataset <name> --start 5860 --limit 50
  python find_bad_images.py --dataset <name> --try-recover   # try alternative decoders; adds 'recovery' column
  python find_bad_images.py --dataset <name> --check-weaviate   # also run weaviate base64 encode check

  HF_TOKEN=<token> python find_bad_images.py --dataset my-org/private-dataset

If --try-recover shows "no", the bytes are likely corrupted; fix by replacing those
images in the dataset. If it shows e.g. "PIL_JPEG", you could use that decoder in your
data loader as a workaround.
"""

import argparse
import csv
import json
import os
import sys
from io import BytesIO, BufferedReader

from PIL import Image


def check_image(
    item: dict,
    idx: int,
    image_column: str = "image",
    check_weaviate: bool = False,
) -> list[str]:
    """
    Run image checks: type, mode (RGB/L), JPEG save; optionally weaviate encode.
    Returns a list of error messages (empty if OK).
    """
    errors = []
    raw_image = item.get(image_column)

    # 1) Type check
    if not isinstance(raw_image, Image.Image):
        errors.append(f"not PIL Image (got {type(raw_image).__name__})")
        return errors

    image = raw_image

    # 2) Mode conversion (JPEG-saveable)
    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
    except Exception as e:
        errors.append(f"convert to RGB: {e}")
        return errors

    # 3) Save to JPEG
    image_stream = BytesIO()
    try:
        image.save(image_stream, format="JPEG", quality=95)
    except Exception as e:
        errors.append(f"JPEG save: {e}")
        return errors

    image_stream.seek(0)
    if image_stream.getbuffer().nbytes == 0:
        errors.append("JPEG save produced empty bytes")
        return errors

    # 4) Optional: Weaviate base64 encode
    if check_weaviate:
        try:
            import weaviate
            buffered_stream = BufferedReader(image_stream)
            weaviate.util.image_encoder_b64(buffered_stream)
        except Exception as e:
            errors.append(f"weaviate encode: {e}")
            return errors

    return errors


def try_recover_image(raw_bytes: bytes) -> tuple[bool, str]:
    """
    Try alternative decoders on raw bytes. Returns (True, decoder_name) if any succeed, else (False, "no").
    """
    buf = BytesIO(raw_bytes)
    for fmt in ("JPEG", "PNG", "BMP", "GIF"):
        try:
            buf.seek(0)
            img = Image.open(buf, formats=[fmt])
            img.load()
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            return True, f"PIL_{fmt}"
        except Exception:
            pass
    try:
        import imageio
        buf.seek(0)
        arr = imageio.v3.imread(buf)
        if arr is not None and arr.size > 0:
            return True, "imageio"
    except Exception:
        pass
    return False, "no"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find rows in a Hugging Face dataset whose image column fails decode/save/encode.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset id (e.g. sagecontinuum/CommonObjectsBench or imdb)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config name (optional)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision/branch (optional)",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of the column containing the image (default: image)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to scan (default: train)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of rows to check (0 = all)",
    )
    parser.add_argument(
        "-o", "--output",
        default="bad_images.csv",
        help="Output CSV path (default: bad_images.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every checked index (not just failures)",
    )
    parser.add_argument(
        "--try-recover",
        action="store_true",
        help="Try alternative decoders on failed images; add 'recovery' column",
    )
    parser.add_argument(
        "--check-weaviate",
        action="store_true",
        help="Also run weaviate base64 encode check (requires weaviate-client)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from datasets.features import Image as HFImageFeature
    except ImportError:
        print("pip install datasets", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.dataset}" + (f" (config={args.config})" if args.config else "") + f" split={args.split} ...")
    ds = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        token=token,
        revision=args.revision,
    )

    if args.image_column not in ds.column_names:
        print(f"Error: column {args.image_column!r} not in dataset. Columns: {ds.column_names}", file=sys.stderr)
        sys.exit(1)

    # Use decode=False so we get raw bytes and decode ourselves (avoids crash on bad bytes)
    if ds.features and args.image_column in ds.features:
        if isinstance(ds.features[args.image_column], HFImageFeature):
            ds = ds.cast_column(args.image_column, HFImageFeature(decode=False))

    total = len(ds)
    print(f"Total rows: {total}, image column: {args.image_column!r}")

    start = max(0, args.start)
    end = total if args.limit <= 0 else min(start + args.limit, total)
    print(f"Checking indices {start} to {end - 1} (count={end - start})")

    non_image_cols = [c for c in ds.column_names if c != args.image_column]
    column_names = ["index", "error", "image_note"] + (["recovery"] if args.try_recover else []) + non_image_cols

    def row_to_report(
        i: int,
        item: dict,
        error: str,
        image_note: str = "",
        recovery: str = "",
    ) -> dict:
        out = {"index": i, "error": error, "image_note": image_note}
        if args.try_recover:
            out["recovery"] = recovery
        for k in non_image_cols:
            v = item.get(k)
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, (list, dict)):
                v = json.dumps(v, default=str)[:500]
            out[k] = "" if v is None else str(v)
        return out

    bad_rows = []
    for i in range(start, end):
        item = dict(ds[i])

        image_cell = item.get(args.image_column)
        raw_bytes = None
        if isinstance(image_cell, dict) and "bytes" in image_cell:
            raw_bytes = image_cell.get("bytes")
        elif isinstance(image_cell, bytes):
            raw_bytes = image_cell
        if raw_bytes is None:
            bad_rows.append(row_to_report(
                i, item,
                "image column has no bytes (path-only or missing)",
                image_note="(no bytes)",
                recovery="n/a" if args.try_recover else "",
            ))
            print(f"  FAIL idx={i} -> no image bytes")
            continue

        try:
            pil_image = Image.open(BytesIO(raw_bytes)).copy()
            pil_image.load()
        except Exception as e:
            recovery_ok, recovery_name = try_recover_image(raw_bytes) if args.try_recover else (False, "")
            bad_rows.append(row_to_report(
                i, item,
                f"decode at load: {e!s}",
                image_note="(decode failed)",
                recovery=recovery_name if args.try_recover else "",
            ))
            msg = f"  FAIL idx={i} -> decode: {e}"
            if args.try_recover:
                msg += f"  [recovery: {recovery_name}]"
            print(msg)
            continue

        item[args.image_column] = pil_image
        errs = check_image(item, i, image_column=args.image_column, check_weaviate=args.check_weaviate)
        if errs:
            bad_rows.append(row_to_report(
                i, item,
                "; ".join(errs),
                image_note="(decode OK, save/encode failed)",
                recovery="n/a" if args.try_recover else "",
            ))
            print(f"  FAIL idx={i} -> {errs}")
        elif args.verbose:
            print(f"  OK   idx={i}")

    print(f"\nTotal failures: {len(bad_rows)}")

    if bad_rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=column_names, extrasaction="ignore")
            w.writeheader()
            w.writerows(bad_rows)
        print(f"Wrote {args.output}")
    else:
        print("No bad images found in the checked range.")


if __name__ == "__main__":
    main()
