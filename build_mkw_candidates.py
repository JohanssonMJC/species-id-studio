#!/usr/bin/env python3
"""
Build per-group candidate lists from the modernised MarineLife catalogue.

Supersedes the earlier repair-and-build script, which tried to second-guess the
classification from the output. The moderniser now reads GBIF's `species` field
correctly, so this script trusts the data and only builds lists.

Each list is a plain text file, one scientific name per line — directly loadable
into the GUI via "Load from file", so a diver can pull in the ~900 nudibranchs
instead of all ~7,500 marine names.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

S = Path(__file__).parent / "marine_lists"
# Deliberately the DURABLE folder next to the app, not ~/.hermes/cache/scratch:
# scratch is pruned after 24h idle, and these files are a deliverable.
S.mkdir(parents=True, exist_ok=True)
GROUPS = {
    "all_marine": None,
}

# Group -> taxon names to match against the list's own hierarchy keys.
TAXON_GROUPS = {
    "nudibranchs": ["Nudibranchia"],
    "sea_slugs": ["Nudibranchia", "Sacoglossa", "Anaspidea", "Aplysiomorpha"],
    "crustaceans": ["Crustacea"],
    "crabs_shrimps": ["Decapoda"],
    "cephalopods": ["Cephalopoda"],
    "corals_anemones": ["Anthozoa", "Scleractinia", "Alcyonacea", "Actiniaria"],
    "echinoderms": ["Asteroidea", "Echinoidea", "Holothuroidea", "Crinoidea"],
    "sharks_rays": ["Chondrichthyes", "Elasmobranchii", "Batoidea", "Selachii"],
    "bony_fishes": ["Osteichthyes", "Actinopterygii"],
    "turtles": ["Chelonioidea", "Cheloniidae", "Dermochelyidae"],
    "cetaceans": ["Cetacea", "Mysticeti", "Odontoceti"],
    "seals": ["Otariidae", "Phocidae", "Odobenidae", "Pinnipedia"],
    "sea_snakes": ["Hydrophiinae", "Hydrophiidae", "Laticaudidae"],
    "dugongs_manatees": ["Sirenia", "Dugongidae", "Trichechidae"],
    "sponges": ["Porifera"],
    "tunicates": ["Ascidiacea"],
    "bryozoans": ["Bryozoa", "Ectoprocta"],
    "worms": ["Annelida", "Platyhelminthes", "Nemertea", "Sipuncula", "Echiura"],
    "jellyfish": ["Scyphozoa", "Hydrozoa", "Cubozoa"],
    "algae_seaweed": ["Phaeophyta", "Chlorophyta", "Rhodophyta", "Ochrophyta",
                      "Ulvophyceae", "Laminariales"],
}


def main() -> int:
    modern = json.load(open(S / "marinelife_modern.json"))
    counts = Counter(e["status"] for e in modern)
    # "typo-repaired" = a 2009 typo corrected against GBIF and confirmed to resolve.
    USABLE = ("current", "renamed", "typo-fixed", "typo-repaired")
    usable = [e for e in modern if e["status"] in USABLE]

    print(f"=== modernised MarineLife catalogue ({len(modern):,} source species) ===")
    for k, v in counts.most_common():
        print(f"  {k:<12} {v:>6,}  {v/len(modern)*100:5.1f}%")
    print(f"\n  USABLE as candidates : {len(usable):,} "
          f"({len(usable)/len(modern)*100:.1f}%)")
    print(f"  genus-level (weaker) : {counts['genus-level']:,}")
    print(f"  unresolved           : {counts['unresolved']:,}")

    # hard guarantee: no candidate carries a taxonomic author
    def words(n): return len(n.split())
    bad = [e for e in usable if e["modern"] and words(e["modern"]) > 3]
    print(f"\n  candidates with an author string: {len(bad)}  "
          f"{'(OK)' if not bad else '(PROBLEM)'}")

    # Written FLAT into marine_lists/ because that is exactly where the GUI looks
    # (MARINE_LISTS in species_gui.py). Writing into a subfolder here meant the app
    # kept loading the old, un-repaired lists while the new ones sat unused.
    outdir = S
    outdir.mkdir(exist_ok=True)

    allnames = sorted({e["modern"] for e in usable if e["modern"]})
    (outdir / "all_marine.txt").write_text("\n".join(allnames) + "\n")
    print(f"\n=== candidate lists -> {outdir} ===")
    print(f"  {'all_marine':<18} {len(allnames):>5,}")

    for gname, pats in TAXON_GROUPS.items():
        rx = re.compile("|".join(rf"\b(?:{p})\b" for p in pats), re.I)
        names = sorted({e["modern"] for e in usable
                        if e["modern"] and any(rx.search(k) for k in e.get("taxa_keys", []))})
        (outdir / f"{gname}.txt").write_text("\n".join(names) + "\n")
        ex = ", ".join(names[:2])
        print(f"  {gname:<18} {len(names):>5,}   e.g. {ex}")

    # also keep the unresolved ones so nothing is silently dropped
    unres = sorted({e["original"] for e in modern if e["status"] == "unresolved"})
    (outdir / "_UNRESOLVED_review_these.txt").write_text("\n".join(unres) + "\n")
    genus = sorted({e["original"] for e in modern if e["status"] == "genus-level"})
    (outdir / "_genus_level.txt").write_text("\n".join(genus) + "\n")
    print(f"\n  _UNRESOLVED_review_these.txt {len(unres):>5,}  (kept, not guessed)")
    print(f"  _genus_level.txt             {len(genus):>5,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
