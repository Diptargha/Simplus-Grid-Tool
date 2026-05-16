"""Run standalone greybox analysis from a config file."""

from __future__ import annotations

import argparse
from pathlib import Path

from simplusgt.export import export_greybox_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SimplusGT greybox analysis")
    parser.add_argument("--config", default="config/greybox_config.json", help="Path to greybox config JSON")
    parser.add_argument("--output", help="Optional output JSON override")
    args = parser.parse_args(argv)
    output = export_greybox_json(None, args.output, config_path=Path(args.config))
    print(f"exported: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
