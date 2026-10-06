"""Convert all non-PDF documents in a directory to PDF using LibreOffice."""

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm

SKIP_SUFFIXES = {".pdf"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in args.input_dir.iterdir()
        if f.is_file() and f.suffix.lower() not in SKIP_SUFFIXES
    ]

    succeeded = []
    failed = []

    for f in tqdm(files):
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--norestore",
                "--convert-to",
                "pdf",
                "--outdir",
                str(args.output_dir),
                str(f),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            succeeded.append(f.name)
        else:
            failed.append((f.name, result.stderr.strip() or result.stdout.strip()))

    # if succeeded:
    #     print("Succeeded:")
    #     for name in succeeded:
    #         print(f"  - {name}")

    if failed:
        for name, err in failed:
            print(f"[!] Failed to convert {name}: {err}")

    print(f"\n{len(succeeded)} succeeded, {len(failed)} failed\n")


if __name__ == "__main__":
    main()
