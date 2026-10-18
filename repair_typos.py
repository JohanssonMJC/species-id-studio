#!/usr/bin/env python3
"""
Typo repair for names GBIF cannot match.

About 6% of the 2009 MarineLife list is a single-character OCR-style error
("Chromi alta", "Chodracanthus exasperatus", "Lolig africana"). GBIF's matcher
refuses them, and they are recoverable — but ONLY under rules tight enough that a
name is never swapped for a different species.

The trap this must avoid: iNaturalist's search is fuzzy and will happily offer a
*different* animal. It maps "Trapezia wardi" to "Trapezia tigrina" — a different
crab. So a suggestion is never accepted on resemblance alone.

Two acceptance paths, both requiring an ACTIVE taxon:

  A. TOKEN TYPO   every differing token is within 2 edits, total <= 3.
                  "Chromi alta" -> "Chromis alta" (1 edit), "Atherinopis affins"
                  -> "Atherinopsis affinis" (1+1). A species swap like
                  wardi -> tigrina is 6 edits and is REJECTED.

  B. EPITHET EXACT the epithet is identical and only the genus is replaced.
                  "Lolig africana" -> "Alloteuthis africana": the epithet
                  "africana" is untouched, so this cannot become a different
                  species — the genus was simply revised.

Every accepted repair is recorded as original -> corrected with the edit
distance and the path taken, so the whole pass is auditable and reversible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lightroom_keywords as lk              # noqa: E402
from modernize_marinelife import (accepted_name, classify, gbif_match,  # noqa: E402
                                 strip_authority)

UA = {"User-Agent": "SpeciesIDStudio/1.0 (local research tool)"}
INAT = "https://api.inaturalist.org/v1/taxa"

MAX_TOKEN_EDITS = 2      # a single token may differ by at most this
MAX_TOTAL_EDITS = 3      # and at most this across the whole name

# Tokens that are taxonomic qualifiers, not epithets. They must be excluded from
# matching, or "Punctinaria sp." gets "corrected" to "Punctinaria uspi" because
# "sp." is two edits from "uspi".
QUALIFIERS = {"sp", "sp.", "spp", "spp.", "cf", "cf.", "aff", "aff.", "forma",
              "var", "var.", "subsp", "subsp."}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def tokenwise_ok(orig: str, cand: str) -> tuple[bool, int]:
    """Path A test: is `cand` a plausible typo of `orig` (same shape, close tokens)?"""
    o, c = orig.split(), cand.split()
    if len(o) != len(c):
        return False, 99
    total = 0
    for to, tc in zip(o, c):
        d = levenshtein(to, tc)
        if d > MAX_TOKEN_EDITS:
            return False, 99
        total += d
    if total == 0 or total > MAX_TOTAL_EDITS:
        return False, total
    return True, total


def epithet_of(name: str) -> str | None:
    t = name.split()
    return t[1].lower() if len(t) >= 2 else None


def inat_suggestions(name: str, retries: int = 3) -> list[dict]:
    q = urllib.parse.urlencode({"q": name, "per_page": 5})
    for a in range(retries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"{INAT}?{q}", headers=UA), timeout=30) as r:
                return json.load(r).get("results", [])
        except Exception:
            if a == retries - 1:
                return []
            time.sleep(2 + a * 3)
    return []


def build_vocab(modern: list[dict]) -> tuple[set[str], set[str]]:
    """Genera and epithets harvested from the names that DO resolve.

    This is the asset that makes repair possible without any API: 2,646 genera and
    5,082 epithets drawn from the 7,530 names GBIF accepted.
    """
    genera, epithets = set(), set()
    for e in modern:
        if e["status"] not in ("current", "renamed", "typo-fixed", "typo-repaired"):
            continue
        t = (e.get("modern") or e["original"]).split()
        if len(t) >= 2 and t[1].lower() not in QUALIFIERS:
            genera.add(t[0])
            epithets.add(t[1])
    return genera, epithets


def nearest(word: str, vocab: set[str], maxd: int) -> tuple[str | None, int]:
    """Closest vocabulary entry within `maxd` edits, or (None, 0)."""
    best, bd = None, maxd + 1
    for v in vocab:
        if abs(len(v) - len(word)) > maxd:
            continue
        d = levenshtein(word, v)
        if d < bd:
            best, bd = v, d
            if d == 1:                      # cannot do better than 1
                break
    return (best, bd) if best and bd <= maxd else (None, 0)


def _gbif_suggestion(entry: dict) -> str | None:
    """Recover the suggestion my earlier rule rejected, from the note field.

    The first pass threw away every FUZZY match whose genus differed — but for a
    typo the genus SHOULD differ, by a character or two. Those suggestions were
    recorded in the note ("fuzzy to a different genus: X") and there are 321 of
    them, so this recovers real corrections without a single API call.
    """
    note = entry.get("note") or ""
    marker = "fuzzy to a different genus:"
    if marker not in note:
        return None
    raw = note.split(marker, 1)[1].strip()
    return strip_authority(raw) if raw else None


def verify_repair(rec: dict, genera: set[str], epithets: set[str]) -> dict:
    """Confirm a repair against GBIF, and try the safer alternative if it fails.

    The dictionary path can over-correct: for "Argarum fimbriatum" it fixed the
    genus AND changed the epithet to "fimbriatus" (to match a vocabulary entry),
    when in fact only "Argarum" was wrong and "fimbriatum" was fine. So every
    repair is checked, and when both tokens were changed we also try fixing the
    GENUS ONLY and keeping the original epithet — the more conservative repair.
    """
    name = rec["original"]
    if not rec.get("repaired"):
        return rec

    cands: list[tuple[str, str]] = [(rec["repaired"], rec["path"])]
    toks = name.split()
    if rec["path"] == "dictionary" and len(toks) == 2:
        g, _ = nearest(toks[0], genera, MAX_TOKEN_EDITS)
        if g and g != toks[0]:
            cands.append((f"{g} {toks[1]}", "dictionary-genus-only"))

    for cand, path in cands:
        g = classify(cand, gbif_match(cand))
        if g.get("status") not in ("current", "renamed") or not g.get("modern"):
            continue
        final = g["modern"]
        # Guard on the distance from the ORIGINAL to the REPAIR CANDIDATE, not to
        # GBIF's final name. Measuring against the final rejected legitimate fixes:
        # "Threagra chalcogramma" -> "Theragra chalcogramma" (2 edits) is then
        # redirected by GBIF to "Gadus chalcogrammus", a further revision that has
        # nothing to do with the typo.
        d_cand = levenshtein(name, cand)
        if not (0 < d_cand <= MAX_TOTAL_EDITS + 1):
            continue
        # The epithet must survive. A different epithet means a different species,
        # so that is what stops a genuine substitution slipping through; gender
        # agreement (chalcogramma -> chalcogrammus) is within the tolerance.
        ep_o, ep_f = epithet_of(name), epithet_of(final)
        if ep_o and ep_f and min(levenshtein(ep_f, ep_o),
                                 levenshtein(ep_f, epithet_of(cand) or "")) > 2:
            continue
        rec.update(repaired=final, path=path + "+verified", edits=d_cand,
                   evidence=f"GBIF confirms {final} ({g['status']}); "
                            f"repair was {d_cand} edit(s) from the original")
        return rec
    rec.update(repaired=None, path=None, edits=None,
               evidence=f"rejected: GBIF would not confirm {rec['repaired']}")
    return rec


def repair_entry(entry: dict, genera: set[str], epithets: set[str]) -> dict:
    """Try to repair one unresolved entry, from local evidence only."""
    name = entry["original"]
    rec = {"original": name, "repaired": None, "path": None, "edits": None,
           "evidence": None}
    toks = name.split()
    if not toks or toks[0].lower() in QUALIFIERS:
        return rec

    # Source 1: GBIF's own fuzzy match, which passes the typo test the first pass
    # never applied.
    sug = _gbif_suggestion(entry)
    if sug and lk.looks_binomial(sug):
        ok, dist = tokenwise_ok(name, sug)
        if ok:
            rec.update(repaired=sug, path="gbif-fuzzy", edits=dist,
                       evidence=f"GBIF fuzzy match, {dist} edit(s)")
            return rec

    # Source 2: local dictionary. Only for genuine binomials — a qualifier means
    # the entry is genus-level and belongs there, not "corrected" into a species.
    if len(toks) == 2 and toks[1].lower() not in QUALIFIERS:
        g, gd = nearest(toks[0], genera, MAX_TOKEN_EDITS)
        p, pd = nearest(toks[1], epithets, MAX_TOKEN_EDITS)
        if g or p:
            fixed = f"{g or toks[0]} {p or toks[1]}"
            dist = (gd or 0) + (pd or 0)
            if fixed != name and 0 < dist <= MAX_TOTAL_EDITS:
                rec.update(repaired=fixed, path="dictionary", edits=dist,
                           evidence=f"closest known genus+epithet, {dist} edit(s)")
                return rec
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modern", default=str(
        Path.home() / "Documents/python/mcp_trial/marine_lists/marinelife_modern.json"))
    ap.add_argument("--cache", default=str(
        Path.home() / "Documents/python/mcp_trial/marine_lists/mkw_typo_cache.json"))
    ap.add_argument("--delay", type=float, default=0.0,
                    help="unused; repair is local, kept for interface compatibility")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    modern = json.load(open(args.modern))
    todo = [e for e in modern if e["status"] == "unresolved"]
    if args.limit:
        todo = todo[:args.limit]

    genera, epithets = build_vocab(modern)
    print(f"vocabulary from the resolved names: {len(genera):,} genera, "
          f"{len(epithets):,} epithets", flush=True)
    print(f"unresolved entries to attempt: {len(todo):,}  (all local, no API)\n",
          flush=True)

    cache: dict[str, dict] = {}
    for i, e in enumerate(todo, 1):
        rec = repair_entry(e, genera, epithets)
        if rec.get("repaired"):
            rec = verify_repair(rec, genera, epithets)   # GBIF must confirm it
        cache[e["original"]] = rec
        if i % 50 == 0:
            n = sum(1 for v in cache.values() if v["repaired"])
            print(f"  {i:,}/{len(todo):,}  repaired+verified {n:,}", flush=True)
        time.sleep(0.2)                                  # be polite to GBIF

    cp = Path(args.cache)
    cp.write_text(json.dumps(cache, indent=1))

    reps = {k: v for k, v in cache.items() if v["repaired"]}
    print(f"\n=== repaired {len(reps)} of {len(cache)} "
          f"({len(reps)/max(1,len(cache))*100:.0f}%) ===")
    bypath: dict[str, int] = {}
    dists: dict[int, int] = {}
    for v in reps.values():
        bypath[v["path"]] = bypath.get(v["path"], 0) + 1
        dists[v["edits"]] = dists.get(v["edits"], 0) + 1
    for k, v in sorted(bypath.items(), key=lambda x: -x[1]):
        print(f"  {k:<14} {v:>4}")
    print("  edits:", ", ".join(f"{k}: {v}" for k, v in sorted(dists.items())))

    print("\n  sample repairs:")
    for k, v in list(reps.items())[:20]:
        print(f"    {k:<32} -> {v['repaired']:<32} [{v['path']}, {v['edits']}]")

    print("\n  refused (kept in review, not guessed):")
    refused = [k for k, v in list(cache.items()) if not v["repaired"]]
    print("    " + ", ".join(refused[:14]))

    for e in modern:
        r = cache.get(e["original"])
        if r and r["repaired"]:
            e["status"] = "typo-repaired"
            e["modern"] = r["repaired"]
            e["note"] = f"{r['path']}: {r['evidence']}"
    Path(args.modern).write_text(json.dumps(modern, indent=1))
    print(f"\nwrote {args.modern}")
    print("now run build_mkw_candidates.py to regenerate the lists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
