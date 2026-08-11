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
INLINE_CODE_PATTERN = re.compile(r"`[^`]*`")
MATH_IN_LINK_PATTERN = re.compile(r"\[[^\]]*\$[^$\]]{1,60}\$[^\]]*]\(")
MATH_IN_EMPHASIS_PATTERN = re.compile(r"(?<![*\w])\*[^*\n]*\$[^$*\n]{1,60}\$[^*\n]*\*(?!\*)")


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


def unrenderable_math(prose: str) -> list[str]:
    r"""Inline math GitHub silently leaves as literal ``$...$``.

    Checked against the published page: of 590 expressions in BENCHMARKS.md,
    571 rendered and 19 did not, and every failure fell into one of the three
    cases below. GitHub needs the opening ``$`` preceded by whitespace or
    start-of-line, and it does not render math inside link text or emphasis.
    ``fixed-$M$``, ``RMSE$(t)$`` and ``$10^{-3}$-$10^{-4}$`` are the shapes
    that actually occurred.

    Delimiters are PAIRED by scanning rather than matched by a regex: a naive
    pattern pairs one expression's closing ``$`` with the next one's opening
    ``$`` and reports six times as many problems as exist.
    """
    found: list[str] = []
    positions = [i for i, c in enumerate(prose) if c == "$"]
    for open_at, close_at in zip(positions[0::2], positions[1::2]):
        before = prose[open_at - 1] if open_at else " "
        if before.isalnum() or before in "-–—_":
            found.append("math will not render, opening $ is glued to "
                         f"{before!r}: {prose[max(0, open_at - 20):close_at + 1].strip()!r}")
    for pattern, why in ((MATH_IN_LINK_PATTERN, "math inside link text will not render"),
                         (MATH_IN_EMPHASIS_PATTERN, "math inside emphasis will not render")):
        for match in pattern.finditer(prose):
            found.append(f"{why}: {match.group(0)[:60]!r}")
    return found


def validate_markdown_integrity() -> int:
    r"""Catch silent LaTeX damage in the Markdown.

    Passing LaTeX through a non-raw Python string or a shell heredoc eats the
    backslash and leaves the control character it named: ``\r`` becomes a
    carriage return, ``\t`` a tab, ``\a`` a bell. None of these are visible in
    an editor, so the damage survives repeated passes over the file -- five
    corrupted ``\rm`` and one ``\rho`` went unnoticed for weeks.

    Also flags inline math split across a line break, which GitHub does not
    render, and display math indented out of column zero, which it also does
    not render.
    """
    problems: list[str] = []
    checked = 0

    for markdown_file in [REPO_ROOT / "README.md", *sorted(BENCHMARK_DIR.rglob("*.md"))]:
        raw = markdown_file.read_bytes()
        text = markdown_file.read_text(encoding="utf-8")
        name = markdown_file.relative_to(REPO_ROOT)
        checked += 1

        lone_cr = sum(1 for k in range(len(raw))
                      if raw[k:k + 1] == b"\r" and raw[k + 1:k + 2] != b"\n")
        if lone_cr:
            problems.append(f"{name}: {lone_cr} lone CR byte(s) -- an eaten LaTeX backslash")

        stray = {c for c in text if ord(c) < 32 and c not in "\n\t"}
        if stray:
            problems.append(f"{name}: control characters {sorted(hex(ord(c)) for c in stray)}")

        # `$USER`, `$1` and the like are shell, not math: blank out code spans
        # and skip fenced blocks before looking at delimiters.
        in_fence = False
        for number, line in enumerate(text.split("\n"), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if line.strip() == "$$" and line != "$$":
                problems.append(f"{name}:{number}: display math indented out of column 0")
                continue
            prose = INLINE_CODE_PATTERN.sub(lambda m: " " * len(m.group(0)), line)
            if prose.count("$") % 2 and prose.strip() != "$$":
                problems.append(f"{name}:{number}: inline math split across a line break")
                continue
            problems.extend(f"{name}:{number}: {why}"
                            for why in unrenderable_math(prose))

    if problems:
        raise RuntimeError("Markdown integrity problems:\n  " + "\n  ".join(problems))
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
    markdown_count = validate_markdown_integrity()
    with tempfile.TemporaryDirectory(prefix="qutip-bundling-smoke-") as temp_dir:
        exercise_commands(Path(temp_dir))
    print(
        "Benchmark smoke test passed: "
        f"{json_count} JSON, {csv_count} CSV, {link_count} local links, "
        f"{markdown_count} Markdown files clean, "
        f"{len(PLOT_COMMANDS)} plot commands, {len(RUNNERS)} protected runners."
    )


if __name__ == "__main__":
    main()
