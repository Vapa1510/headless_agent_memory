from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

INCLUDE_ROOTS = (
    "src",
    "tests",
    "eval",
    "scripts/package_submission.py",
    "benchmarks/run_benchmarks.py",
    "benchmarks/results.json",
    "brief/Agentic_Memory_Brief.pdf",
    "brief/brief.md",
    "brief/fig1_architecture.png",
    "brief/fig2_measurements.png",
    "brief/generate_brief.py",
    "brief/figures.py",
    "brief/build_pdf.py",
    "brief/brief_content.py",
    "pyproject.toml",
    "README.md",
    "ARCHITECTURE.md",
    "REVIEW_RESPONSE.md",
    ".gitignore",
)

FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "env"}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3")
FIXED_ZIP_DATE = (2026, 10, 15, 0, 0, 0)


def _is_clean_relpath(rel: Path) -> bool:
    if any(part in FORBIDDEN_PARTS for part in rel.parts):
        return False
    if rel.as_posix().startswith(("benchmarks/data/", "eval/data/", "dist/")):
        return False
    if rel.name == ".env" or rel.name.startswith(".env."):
        return False
    return not (rel.name.endswith(FORBIDDEN_SUFFIXES) or rel.name == "_raw_brief.pdf")


def collect_submission_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for entry in INCLUDE_ROOTS:
        target = repo_root / entry
        if not target.exists():
            raise FileNotFoundError(f"Required submission path missing: {entry}")
        if target.is_file():
            rel = target.relative_to(repo_root)
            if _is_clean_relpath(rel):
                files.append(target)
        else:
            for p in target.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(repo_root)
                    if _is_clean_relpath(rel):
                        files.append(p)
    files.sort(key=lambda p: p.relative_to(repo_root).as_posix())
    return files


def build_submission_zip(repo_root: Path, out_zip: Path) -> tuple[Path, int, str]:
    files = collect_submission_files(repo_root)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            rel_posix = path.relative_to(repo_root).as_posix()
            info = zipfile.ZipInfo(filename=rel_posix, date_time=FIXED_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())

    digest = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    return out_zip, len(files), digest


def verify_submission_zip(out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
        for name in names:
            rel = Path(name)
            if not _is_clean_relpath(rel):
                raise AssertionError(f"Forbidden cache or DB artifact inside zip: {name}")
        with tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            clean_env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=td,
                env=clean_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Fresh-unpack pytest failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
                )
            eval_proc = subprocess.run(
                [sys.executable, "eval/run_live_eval.py", "--mode", "offline"],
                cwd=td,
                env=clean_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )
            if eval_proc.returncode != 0:
                raise RuntimeError(
                    f"Fresh-unpack standalone eval failed ({eval_proc.returncode}):\n{eval_proc.stdout}\n{eval_proc.stderr}"
                )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build and verify clean submission zip.")
    parser.add_argument(
        "--output",
        default=str(repo_root / "dist" / "agentic-memory-submission.zip"),
        help="Destination zip path.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip running pytest inside unpacked archive.",
    )
    args = parser.parse_args()
    out_zip, file_count, sha = build_submission_zip(repo_root, Path(args.output))
    if not args.skip_pytest:
        verify_submission_zip(out_zip)
    print(f"Packaged {file_count} clean files -> {out_zip} (sha256={sha})")


if __name__ == "__main__":
    main()
