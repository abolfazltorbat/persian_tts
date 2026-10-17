"""Explicit one-time download; runtime loads exclusively from local files."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["0.6", "1.7"], default="0.6")
    parser.add_argument("--revision", default="main", help="Hugging Face commit or revision")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    repo = f"Qwen/Qwen3-TTS-12Hz-{args.size}B-Base"
    target = Path(__file__).resolve().parents[1] / "model" / f"qwen3-tts-{args.size}b"
    revision = HfApi().model_info(repo, revision=args.revision).sha
    snapshot_download(repo, revision=revision, local_dir=target)
    (target / "download.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))
    print(f"Downloaded {repo} at {revision} to {target}")


if __name__ == "__main__":
    main()
