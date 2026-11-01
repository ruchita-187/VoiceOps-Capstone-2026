import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VOICE_SCRIPT = PROJECT_ROOT / "src" / "step4_voice_batch.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="Path to a folder containing customer audio files."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not input_path.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {input_path}")

    if not VOICE_SCRIPT.is_file():
        raise FileNotFoundError(f"Batch script not found: {VOICE_SCRIPT}")

    subprocess.run(
        [sys.executable, str(VOICE_SCRIPT), str(input_path)],
        cwd=PROJECT_ROOT,
        check=True
    )


if __name__ == "__main__":
    main()
    