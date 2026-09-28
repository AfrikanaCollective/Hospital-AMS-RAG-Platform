"""Pre-fetch the layout parser's model weights into the local caches
(ARCH-044, PRD-113; LAYOUT-INGESTION-PROPOSAL.md §11).

Docling's layout and TableFormer weights (Hugging Face cache, the
`hf-model-cache` volume) and the OCR engine's weights are downloaded on
first use. Run this once, with network, so that ingestion itself never
reaches out: initialising the converter for the configured
`INGEST_OCR_ENGINE` pulls everything the pipeline needs.

Usage (inside the api or worker container):
    python -m scripts.fetch_layout_models [--engine rapidocr|easyocr|tesseract]
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=get_settings().ingest_ocr_engine)
    args = parser.parse_args(argv)

    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415

    from app.ingestion.layout.docling_adapter import _converter  # noqa: PLC0415

    converter = _converter(args.engine, get_settings().ingest_images_scale)
    converter.initialize_pipeline(InputFormat.PDF)
    print(f"[fetch-layout-models] layout, table and {args.engine} OCR models are cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
