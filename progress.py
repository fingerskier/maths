#!/usr/bin/env python3
"""
Mathematics Level-Up Progress Checker

Scans coursework directories for problem files and reports completion status.
A problem is considered "complete" when it has no remaining `raise NotImplementedError`
statements and all functions have implementations.

Usage:
    python progress.py              # Full summary
    python progress.py --level 0    # Check specific level
    python progress.py --verbose    # Show per-file details
"""

import argparse
import os
import re
import sys
from pathlib import Path

COURSEWORK_DIR = Path(__file__).parent / "coursework"

LEVEL_NAMES = {
    0: "Foundations of Mathematics",
    1: "Discrete Mathematics and Combinatorics",
    2: "Calculus and Analysis",
    3: "Optimization",
    4: "Probability and Statistics",
    5: "Numerical and Scientific Computing",
    6: "Machine Learning and Data Science",
    7: "Advanced Simulation and Data-Integrated Methods",
    8: "Applied Domains",
}


def find_level_dirs():
    """Find all level directories, sorted by level number."""
    levels = {}
    for d in sorted(COURSEWORK_DIR.iterdir()):
        if d.is_dir():
            match = re.match(r"(\d+)-", d.name)
            if match:
                levels[int(match.group(1))] = d
    return levels


def scan_problem_file(filepath):
    """
    Scan a single problem file.

    Returns:
        dict with keys: path, total_functions, implemented, not_implemented
    """
    with open(filepath, "r") as f:
        source = f.read()

    # Count functions defined
    func_defs = re.findall(r"^def (\w+)\(", source, re.MULTILINE)
    # Count NotImplementedError raises
    not_impl = len(re.findall(r"raise NotImplementedError", source))

    total = len(func_defs)
    implemented = total - not_impl

    return {
        "path": filepath,
        "total_functions": total,
        "implemented": max(implemented, 0),
        "not_implemented": not_impl,
    }


def scan_level(level_dir):
    """Scan all problem files in a level directory."""
    results = []
    for problem_file in sorted(level_dir.rglob("problems/*.py")):
        if problem_file.name.startswith("test_"):
            continue
        results.append(scan_problem_file(problem_file))
    return results


def print_level_summary(level_num, level_dir, results, verbose=False):
    """Print summary for a single level."""
    name = LEVEL_NAMES.get(level_num, "Unknown")
    total_funcs = sum(r["total_functions"] for r in results)
    implemented = sum(r["implemented"] for r in results)
    num_files = len(results)

    if total_funcs == 0:
        pct = 0
    else:
        pct = (implemented / total_funcs) * 100

    # Level status
    if pct == 100 and num_files > 0:
        status = "COMPLETE"
    elif pct > 0:
        status = "IN PROGRESS"
    else:
        status = "NOT STARTED"

    bar_len = 20
    filled = int(bar_len * pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(f"\nLevel {level_num}: {name}")
    print(f"  [{bar}] {pct:5.1f}%  ({implemented}/{total_funcs} functions)  [{status}]")
    print(f"  {num_files} problem files")

    if verbose:
        for r in results:
            rel = os.path.relpath(r["path"], COURSEWORK_DIR)
            mark = "ok" if r["not_implemented"] == 0 else f"{r['not_implemented']} TODO"
            print(f"    {rel}: {r['implemented']}/{r['total_functions']} [{mark}]")

    return pct == 100 and num_files > 0


def main():
    parser = argparse.ArgumentParser(description="Check coursework progress")
    parser.add_argument("--level", type=int, help="Check a specific level")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file details")
    args = parser.parse_args()

    levels = find_level_dirs()

    if args.level is not None:
        if args.level not in levels:
            print(f"Level {args.level} not found.")
            sys.exit(1)
        results = scan_level(levels[args.level])
        print_level_summary(args.level, levels[args.level], results, args.verbose)
        return

    print("=" * 60)
    print("  MATHEMATICS LEVEL-UP PROGRESS")
    print("=" * 60)

    completed_levels = []
    for level_num in sorted(levels.keys()):
        results = scan_level(levels[level_num])
        complete = print_level_summary(level_num, levels[level_num], results, args.verbose)
        if complete:
            completed_levels.append(level_num)

    print("\n" + "=" * 60)
    total_levels = len(levels)
    print(f"  Levels completed: {len(completed_levels)}/{total_levels}")

    # Check gating: can only "unlock" level N if level N-1 is complete
    max_unlocked = 0
    for i in sorted(levels.keys()):
        if i == 0 or (i - 1) in completed_levels:
            max_unlocked = i
        else:
            break

    print(f"  Current level unlocked: {max_unlocked}")
    print("=" * 60)


if __name__ == "__main__":
    main()
