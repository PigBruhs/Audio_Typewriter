from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_APP_PATH = ROOT / "apps" / "api"
if str(API_APP_PATH) not in sys.path:
    sys.path.insert(0, str(API_APP_PATH))

from app.services.asr_service import ASRService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or warm up faster-whisper models.")
    parser.add_argument("--model-tier", default="large", help="tiny/base/small/medium/large or explicit name")
    parser.add_argument("--language", default="en", help="ASR language, currently en only")
    parser.add_argument("--model-name", default=None, help="Optional explicit model name or HF repo id")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    service = ASRService()
    result = service.download_model(
        model_tier=args.model_tier,
        language=args.language,
        model_name=args.model_name,
    )
    print(f"model_name={result.model_name}")
    print(f"status={result.status}")
    print(f"device_used={result.device_used}")
    print(f"compute_type={result.compute_type}")
    print(f"cache_dir={result.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

