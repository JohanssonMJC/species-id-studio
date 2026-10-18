#!/usr/bin/env python3
"""Build the marine candidate lists from a MarineLife Keywords export you own.

    python make_marine_lists.py "/path/to/MarineLife Keywords 1.0.txt"

Why this exists: the derived `marine_lists/*.txt` files are NOT shipped in this
repository — they are downstream of a commercial keyword product. If you own a copy,
this script regenerates exactly the same lists locally in four stages:

  1. parse      the export into species records      (offline, seconds)
  2. modernise  every name against the GBIF backbone (network, ~45 min uncached)
  3. repair     the residual single-character typos  (network, a few minutes)
  4. emit       one .txt per group for the GUI       (offline, seconds)

Stages 2 and 3 cache every lookup, so an interrupted run resumes where it stopped and
a second run costs nothing.

Nothing here redistributes the source list: your copy stays on your machine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_source(path: Path, workdir: Path) -> Path:
    """Accept the .txt or the .zip it came in."""
    if path.is_dir():
        raise SystemExit(f"{path} is a directory — point at the .txt or .zip file")
    if path.suffix.lower() == ".zip":
        out = workdir / "source"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")
                     and not Path(n).name.startswith("._")]
        if not names:
            raise SystemExit(f"no .txt inside {path.name}")
        z = zipfile.ZipFile(path)
        target = out / Path(names[0]).name
        with z.open(names[0]) as src, open(target, "wb") as dst:
            dst.write(src.read())
        print(f"  extracted {names[0]} -> {target}")
        return target
    return path


def run(stage: str, cmd: list[str]) -> None:
    print(f"\n=== {stage} ===", flush=True)
    print("  " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"{stage} failed (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate the marine candidate lists from your own keyword export.")
    ap.add_argument("source", type=Path,
                    help="the MarineLife Keywords export (.txt), or the .zip you got it in")
    ap.add_argument("--out", type=Path, default=HERE / "marine_lists",
                    help="where the .txt lists are written (default: ./marine_lists)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="GBIF lookup cache to reuse/create (default: <out>/mkw_gbif_cache.json)")
    ap.add_argument("--delay", type=float, default=0.2,
                    help="seconds between GBIF lookups (default 0.2; be polite)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop modernisation after N names (0 = all) — for a smoke test")
    ap.add_argument("--skip-modernise", action="store_true",
                    help="stop after parsing, to check the export was read correctly")
    a = ap.parse_args()

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    cache: Path = a.cache or (out / "mkw_gbif_cache.json")
    entries = out / "marinelife_entries.json"
    modern = out / "marinelife_modern.json"
    py = sys.executable

    src = find_source(a.source, out)
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")

    # ---- 1. parse -----------------------------------------------------------------
    print("\n=== 1. parse the export ===")
    import lightroom_keywords as lk
    text = src.read_text(encoding="utf-8", errors="replace")
    tree = lk.parse(text)
    recs = lk.species_entries(tree)
    if not recs:
        raise SystemExit("the export parsed to zero species — wrong file, or an "
                         "unknown keyword-export shape")
    entries.write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    print(f"  {len(recs):,} species records -> {entries.name}")
    print(f"  e.g. {recs[0]['scientific']}  ({', '.join(recs[0]['common'][:2]) or 'no common name'})")
    if a.skip_modernise:
        print("\nstopping here (--skip-modernise)")
        return 0

    # ---- 2. modernise -------------------------------------------------------------
    if cache.is_file():
        print(f"\n  reusing {cache.name} — cached lookups are free")
    else:
        print(f"\n  no cache yet: {len(recs):,} GBIF lookups at {a.delay}s each "
              f"≈ {len(recs) * a.delay / 60:.0f} min. Interrupt safely; it resumes.")
    run("2. modernise names against GBIF", [
        py, str(HERE / "modernize_marinelife.py"),
        "--src", str(entries), "--cache", str(cache), "--out", str(modern),
        "--delay", str(a.delay)] + (["--limit", str(a.limit)] if a.limit else []))

    # ---- 3. repair typos ----------------------------------------------------------
    run("3. repair residual typos (strict, verified)", [
        py, str(HERE / "repair_typos.py"),
        "--modern", str(modern), "--cache", str(out / "mkw_typo_cache.json")])

    # ---- 4. emit ------------------------------------------------------------------
    run("4. emit per-group lists", [
        py, str(HERE / "build_mkw_candidates.py"), "--dir", str(out)])

    print("\n=== done ===")
    lists = sorted(p for p in out.glob("*.txt"))
    total = sum(1 for p in lists if not p.name.startswith("_"))
    print(f"  {len(lists)} files in {out} ({total} loadable groups)")
    for p in lists[:6]:
        n = sum(1 for l in p.read_text().splitlines() if l.strip())
        print(f"    {p.name:<28} {n:>5} names")
    if len(lists) > 6:
        print(f"    … and {len(lists) - 6} more")
    print("\n  Launch the app and use “Marine list…” to load them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
