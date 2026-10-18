#!/usr/bin/env python3
"""
Modernise the MarineLife Keywords taxonomy (v1.0, April 2009) against GBIF.

WHY GBIF AND NOT iNATURALIST SEARCH: iNaturalist's taxa search returns a fuzzy
"nearest" match, which is not safe to accept — it will happily offer a different,
merely similar-looking species. GBIF's species/match endpoint is built for exactly
this job and returns both matchType (EXACT/FUZZY/HIGHERRANK/NONE) and status
(ACCEPTED/SYNONYM) plus acceptedScientificName, so a rename can be applied on
evidence rather than on resemblance.

DECISION RULES (deliberately conservative):
  EXACT   + ACCEPTED  -> keep the name unchanged
  EXACT   + SYNONYM   -> apply acceptedScientificName (an authoritative rename)
  FUZZY   + same genus-> accept but FLAG (usually a one-letter typo:
                         "Trincheasia virens" -> "Trinchesia virens")
  FUZZY   + diff genus-> leave unresolved; a differing genus is a red flag
  HIGHERRANK          -> genus-level entry ("Cuthona sp."); record the genus
  NONE                -> unresolved (probably a genuine typo in the source)

Resumable: every result is cached to disk and flushed periodically, so the job can
be interrupted or re-run without redoing work.

Usage: modernize_marinelife.py [--src entries.json] [--cache cache.json]
                               [--out modern.json] [--delay 0.2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lightroom_keywords as lk          # noqa: E402

GBIF = "https://api.gbif.org/v1/species/match"
UA = {"User-Agent": "SpeciesIDStudio/1.0 (local offline research tool)"}


def gbif_match(name: str, timeout: int = 30) -> dict:
    q = urllib.parse.urlencode({"name": name, "strict": "false"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"{GBIF}?{q}", headers=UA),
                    timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(15 * (attempt + 1))
                continue
            return {"_error": f"HTTP {e.code}"}
        except Exception as e:
            if attempt == 4:
                return {"_error": f"{type(e).__name__}: {e}"}
            time.sleep(3 * (attempt + 1))
    return {"_error": "retries exhausted"}


def genus_of(name: str) -> str:
    return name.split()[0] if name.split() else ""


def strip_authority(name: str) -> str:
    """Reduce 'Genus species Author, 1963' to 'Genus species'.

    GBIF's `scientificName` and `acceptedScientificName` carry the taxonomic
    AUTHORITY; `species` and `canonicalName` do not. Candidates must be bare
    binomials — "Porphyra perforata J.Agardh" is not a name BioCLIP can match.
    Intraspecific qualifiers are kept: "X y forma z".
    """
    toks = name.split()
    if len(toks) <= 2:
        return name
    out = toks[:2]
    for t in toks[2:]:
        if t[:1].isupper() or t.startswith("("):   # authorship begins here
            break
        out.append(t)                              # forma / var. / subsp.
    return " ".join(out)


def accepted_name(d: dict) -> str:
    """The ACCEPTED binomial, free of authorship.

    Field priority matters enormously. GBIF returns the accepted species in
    `species`; `scientificName` is the name as MATCHED and still carries the old
    combination plus its authority. Reading scientificName first silently keeps
    obsolete names: for Chromodoris binza it yields "Chromodoris binza
    Ev.Marcus & Er.Marcus, 1963" while the accepted name is Felimida binza.
    """
    for key in ("species", "acceptedScientificName", "canonicalName", "scientificName"):
        v = str(d.get(key) or "").strip()
        if v:
            return strip_authority(v)
    return ""


def classify(name: str, d: dict) -> dict:
    """Turn a GBIF match into a decision. See module docstring for the rules."""
    if not d or d.get("_error"):
        return {"status": "error", "modern": None, "note": (d or {}).get("_error", "")}
    mt = d.get("matchType")
    st = d.get("status")
    acc = accepted_name(d)
    raw = {k: d.get(k) for k in ("matchType", "status", "species", "genus",
                                 "canonicalName", "acceptedUsageKey", "usageKey",
                                 "rank", "confidence")}
    base = {"_gbif": raw}
    if mt == "EXACT" and st == "ACCEPTED":
        return {**base, "status": "current", "modern": name, "note": ""}
    if mt == "EXACT" and st == "SYNONYM" and acc:
        if acc.lower() == strip_authority(name).lower():
            # GBIF calls it a synonym but the accepted binomial is the same name:
            # an authorship/combination nuance, not a rename. Keep the clean name.
            return {**base, "status": "current", "modern": strip_authority(name),
                    "note": "synonym of itself (authorship only)"}
        return {**base, "status": "renamed", "modern": acc, "note": "gbif synonym"}
    if mt == "FUZZY" and acc:
        if genus_of(acc).lower() == genus_of(name).lower():
            return {**base, "status": "typo-fixed", "modern": acc,
                    "note": "fuzzy, same genus"}
        return {**base, "status": "unresolved", "modern": None,
                "note": f"fuzzy to a different genus: {acc}"}
    if mt == "HIGHERRANK":
        return {**base, "status": "genus-level", "modern": None,
                "note": f"genus only: {d.get('genus') or genus_of(name)}"}
    return {**base, "status": "unresolved", "modern": None,
            "note": f"matchType={mt} status={st}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(
        HERE / "marine_lists" / "marinelife_entries.json"))
    ap.add_argument("--cache", default=str(
        HERE / "marine_lists" / "mkw_gbif_cache.json"))
    ap.add_argument("--out", default=str(
        HERE / "marine_lists" / "marinelife_modern.json"))
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="stop after N (0 = all)")
    ap.add_argument("--redo-status", default="",
                    help="comma-separated statuses to RE-resolve (e.g. "
                         "'renamed,typo-fixed'). Use after fixing the classification "
                         "rules: only those names were decided by the old logic.")
    args = ap.parse_args()

    entries = json.load(open(args.src))
    names = [e["scientific"] for e in entries]
    cache: dict[str, dict] = {}
    cp = Path(args.cache)
    if cp.exists():
        try:
            cache = json.loads(cp.read_text())
        except Exception:
            cache = {}
    if args.redo_status:
        want = {s.strip() for s in args.redo_status.split(",") if s.strip()}
        todo = [n for n in names if (cache.get(n) or {}).get("status") in want]
        for n in todo:
            cache.pop(n, None)              # force a fresh lookup
        print(f"re-resolving {len(todo):,} names previously classified as {sorted(want)}")
    else:
        todo = [n for n in names if n not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(names):,} species; {len(cache):,} already cached; "
          f"{len(todo):,} to look up  (delay {args.delay}s -> "
          f"~{len(todo)*args.delay/60:.0f} min)", flush=True)

    t0 = time.time()
    for i, n in enumerate(todo, 1):
        cache[n] = classify(n, gbif_match(n))
        time.sleep(args.delay)
        if i % 25 == 0 or i == len(todo):
            cp.write_text(json.dumps(cache))            # flush: job is resumable
            rate = i / max(1e-6, time.time() - t0)
            eta = (len(todo) - i) / max(1e-6, rate) / 60
            print(f"  {i:,}/{len(todo):,}  {rate:.1f}/s  eta ~{eta:.0f} min", flush=True)

    cp.write_text(json.dumps(cache))

    # ---- assemble the modernised catalogue ---------------------------------
    by_sci = {e["scientific"]: e for e in entries}
    modern, counts = [], {}
    for name, res in cache.items():
        e = by_sci.get(name)
        if e is None:
            continue
        st = res.get("status", "error")
        counts[st] = counts.get(st, 0) + 1
        modern.append({
            "original": name,
            "modern": res.get("modern"),
            "status": st,
            "note": res.get("note", ""),
            "common": e.get("common", []),
            "path": e.get("path", []),
            "taxa_keys": lk.path_taxon_keys(e),
        })
    Path(args.out).write_text(json.dumps(modern, indent=1))

    done = len(cache)
    print(f"\n=== result over {done:,} species ===")
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k:<12} {counts[k]:>6,}  {counts[k]/max(1,done)*100:5.1f}%")
    usable = sum(v for k, v in counts.items() if k in ("current", "renamed", "typo-fixed"))
    print(f"\n  USABLE as candidates now : {usable:,} ({usable/max(1,done)*100:.1f}%)")
    print(f"  genus-level (weaker)     : {counts.get('genus-level',0):,}")
    print(f"  unresolved (need review) : {counts.get('unresolved',0):,}")
    print(f"  errors                   : {counts.get('error',0):,}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
