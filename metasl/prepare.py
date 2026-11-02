"""Prepare an offline model configuration and Meta structure manifest."""

import argparse
import json
from pathlib import Path

from metasl.skeleton import DTYPES, prepare_bundle, require_torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    require_torch()
    print(json.dumps(prepare_bundle(args.model_id, args.dtype, args.bundle,
                                  local_files_only=args.local_files_only)))


if __name__ == "__main__":
    main()
