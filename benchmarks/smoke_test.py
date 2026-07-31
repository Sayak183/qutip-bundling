"""Exercise the benchmark newcomer workflow without running any solvers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
DATA_DIR = BENCHMARK_DIR / "data"

PLOT_COMMANDS = (
    ("plot_accuracy_vs_M.py", "--system", "spin_chain"),
    ("plot_accuracy_vs_M.py",),
    ("plot_cost_scaling.py",),
    ("plot_frontier.py",),
    ("plot_isocost_vs_dim.py",),
)
RUNNERS = (
    "run_accuracy_vs_M.py",
    "run_cost_scaling.py",
    "run_frontier.py",
    "run_isocost_vs_dim.py",
    "run_method_comparison.py",
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def run_script(
    args: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    expect_success: bool = True,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    """Run a benchmark script and enforce its expected exit status."""
    print(f"$ {' '.join(args)}")
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_DIR / args[0]), *args[1:]],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if (result.returncode == 0) != expect_success:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        expectation = "succeed" if expect_success else "fail"
        raise RuntimeError(
            f"{' '.join(args)} should {expectation}; "
            f"exit status was {result.returncode}"
        )
    return result


def validate_saved_data() -> tuple[int, int]:
    """Parse every committed benchmark JSON and CSV file."""
    json_files = sorted(BENCHMARK_DIR.rglob("*.json"))
    for path in json_files:
        with path.open(encoding="utf-8") as stream:
            json.load(stream)

    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    for path in csv_files:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            if not next(rows, None) or next(rows, None) is None:
                raise RuntimeError(f"CSV has no header or data row: {path}")
    return len(json_files), len(csv_files)


def link_target(raw_target: str) -> str:
    """Extract the local path portion of a Markdown link target."""
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    return unquote(target.split("#", maxsplit=1)[0])


def validate_local_links() -> int:
    """Ensure local Markdown links and embedded images resolve."""
    markdown_files = [REPO_ROOT / "README.md"]
    markdown_files.extend(sorted(BENCHMARK_DIR.rglob("*.md")))
    missing: list[str] = []
    checked = 0

    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = link_target(raw_target)
            if not target:
                continue
            checked += 1
            if not (markdown_file.parent / target).resolve().exists():
                missing.append(
                    f"{markdown_file.relative_to(REPO_ROOT)} -> {target}"
                )

    if missing:
        raise RuntimeError("Missing local Markdown targets:\n" + "\n".join(missing))
    return checked


def json_hashes(root: Path) -> dict[Path, str]:
    """Return SHA-256 hashes for JSON files below *root*."""
    return {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def exercise_commands(scratch_dir: Path) -> None:
    """Run plots and CLI safety paths against a temporary data copy."""
    scratch_data = scratch_dir / "data"
    shutil.copytree(DATA_DIR, scratch_data)
    before = json_hashes(scratch_data)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(scratch_dir / ".matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for command in PLOT_COMMANDS:
        run_script(command, cwd=scratch_dir, env=env)

    for runner in RUNNERS:
        run_script((runner, "--help"), cwd=scratch_dir, env=env)
        failure = run_script(
            (runner,), cwd=scratch_dir, env=env, expect_success=False
        )
        message = failure.stdout + failure.stderr
        if "--system" not in message or "--all" not in message:
            raise RuntimeError(f"{runner} did not explain its required scope")

    documented_preview = run_script(
        (
            "run_frontier.py",
            "--system",
            "spin_chain",
            "--dims",
            "64",
            "--dry-run",
        ),
        cwd=scratch_dir,
        env=env,
    )
    safety_message = "no solvers ran and no files were written"
    if safety_message not in documented_preview.stdout:
        raise RuntimeError("The documented frontier preview was not a dry run")

    for runner in RUNNERS:
        preview = run_script(
            (runner, "--all", "--dry-run"), cwd=scratch_dir, env=env
        )
        if safety_message not in preview.stdout:
            raise RuntimeError(f"{runner} did not confirm dry-run safety")

    after = json_hashes(scratch_data)
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [str(path) for path in changed if before.get(path) != after.get(path)]
        raise RuntimeError("Dry runs changed JSON: " + ", ".join(changed))


def main() -> None:
    json_count, csv_count = validate_saved_data()
    link_count = validate_local_links()
    with tempfile.TemporaryDirectory(prefix="qutip-bundling-smoke-") as temp_dir:
        exercise_commands(Path(temp_dir))
    print(
        "Benchmark smoke test passed: "
        f"{json_count} JSON, {csv_count} CSV, {link_count} local links, "
        f"{len(PLOT_COMMANDS)} plot commands, {len(RUNNERS)} protected runners."
    )


if __name__ == "__main__":
    main()
