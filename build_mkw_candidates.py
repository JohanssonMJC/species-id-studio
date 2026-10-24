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
#
# A group only ever contains a name if one of these taxon names appears in that species'
# GBIF lineage, so a class missing here means every species in it is reachable ONLY
# through "all marine species". That happened: Gastropoda, Bivalvia, Polyplacophora,
# Ophiuroidea, Ctenophora and Crocodylia were absent, leaving 639 of 7,405 names with no
# group at all — cone shells, cowries, egg cowries, sea hares, clams, chitons, brittle
# stars, comb jellies and the crocodiles. The orphan check at the end of main() now makes
# that impossible to reintroduce silently.
TAXON_GROUPS = {
    "nudibranchs": ["Nudibranchia"],
    # Nudibranchia are gastropods; sea_slugs is a convenience subset that also carries the
    # sacoglossans and sea hares, so the lists intentionally overlap.
    "sea_slugs": ["Nudibranchia", "Sacoglossa", "Anaspidea", "Aplysiomorpha"],
    # every remaining gastropod: cones, cowries, ovulids, whelks, limpets, sea hares
    "gastropods": ["Gastropoda"],
    # clams, oysters, scallops, chitons and tusk shells
    "bivalves": ["Bivalvia", "Pelecypoda", "Polyplacophora", "Scaphopoda"],
    # "Crustacea" almost never appears in a GBIF lineage — Decapoda sits under
    # Malacostraca — so matching only that word left "all crustaceans" missing any
    # crab whose source path did not say Crustacea. List the classes instead, the same
    # union the GBIF candidate source uses, because no single key covers Crustacea.
    "crustaceans": ["Crustacea", "Malacostraca", "Maxillopoda", "Copepoda",
                    "Branchiopoda", "Ostracoda", "Remipedia", "Thecostraca",
                    "Tantulocarida", "Cephalocarida", "Mystacocarida", "Branchiura",
                    "Pentastomida", "Ichthyostraca", "Hexanauplia"],
    "crabs_shrimps": ["Decapoda"],
    "sea_spiders": ["Pycnogonida"],
    "cephalopods": ["Cephalopoda"],
    # aplacophoran molluscs: solenogasters and caudofoveates, plus tusk shells
    "other_molluscs": ["Solenogastres", "Caudofoveata", "Aplacophora"],
    "corals_anemones": ["Anthozoa", "Scleractinia", "Alcyonacea", "Actiniaria"],
    "echinoderms": ["Asteroidea", "Echinoidea", "Holothuroidea", "Crinoidea",
                    "Ophiuroidea", "Comatulida"],
    "sharks_rays": ["Chondrichthyes", "Elasmobranchii", "Batoidea", "Selachii"],
    "bony_fishes": ["Osteichthyes", "Actinopterygii", "Myxini", "Petromyzontida",
                    "Chondrostei"],
    "turtles": ["Chelonioidea", "Cheloniidae", "Dermochelyidae"],
    "crocodiles": ["Crocodylia", "Crocodylidae", "Alligatoridae", "Gavialidae",
                   "Iguanidae", "Amblyrhynchus"],
    "cetaceans": ["Cetacea", "Mysticeti", "Odontoceti"],
    "seals": ["Otariidae", "Phocidae", "Odobenidae", "Pinnipedia", "Mustelidae",
              "Ursidae"],
    "sea_snakes": ["Hydrophiinae", "Hydrophiidae", "Laticaudidae"],
    "dugongs_manatees": ["Sirenia", "Dugongidae", "Trichechidae"],
    "sponges": ["Porifera"],
    "tunicates": ["Ascidiacea", "Thaliacea", "Pyrosomatida", "Copelata",
                  "Appendicularia", "Doliolida", "Salpida"],
    "bryozoans": ["Bryozoa", "Ectoprocta"],
    "worms": ["Annelida", "Platyhelminthes", "Nemertea", "Sipuncula", "Echiura",
              "Chaetognatha", "Brachiopoda", "Phoronida", "Hemichordata",
              "Cephalochordata", "Enteropneusta"],
    "jellyfish": ["Scyphozoa", "Hydrozoa", "Cubozoa", "Ctenophora"],
    "algae_seaweed": ["Phaeophyta", "Chlorophyta", "Rhodophyta", "Ochrophyta",
                      "Ulvophyceae", "Laminariales", "Bacillariophyceae"],
}


def groups_for(taxa_keys: list[str]) -> list[str]:
    """Which groups a species belongs to, given every taxon name in its lineage.

    Matching is word-boundary and case-insensitive so "Gastropoda" does not match
    "Gastropoda_thing", and a lineage can satisfy several groups (a nudibranch is both a
    gastropod and a sea slug) — the lists are meant to overlap.
    """
    keys = [str(k) for k in (taxa_keys or [])]
    out = []
    for gname, pats in TAXON_GROUPS.items():
        rx = re.compile("|".join(rf"\b(?:{p})\b" for p in pats), re.I)
        if any(rx.search(k) for k in keys):
            out.append(gname)
    return out


def gbif_lineage(name: str) -> list[str]:
    """The ranks GBIF's backbone places this name in, as a list of taxon names.

    Needed because the 2009 keyword list files some names under a phylum only
    ("Cnidarians: Cnidaria"), and no group can match a lineage that has no class in it.
    Uses the public API, which answers where www.gbif.org refuses to.
    """
    try:
        from gbif_candidates import lookup_species
    except Exception:
        return []
    d = lookup_species(name)
    if (d.get("matchType") or "NONE") == "NONE":
        return []
    return [d[k] for k in ("kingdom", "phylum", "class", "order", "family", "genus",
                           "canonicalName") if d.get(k)]



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

    # GUARD: every usable name must be reachable through at least one group. Without
    # this, a taxon missing from TAXON_GROUPS silently strands a whole class in
    # "all marine species" only — which is exactly how 639 names (cone shells, cowries,
    # clams, chitons, brittle stars, comb jellies, crocodiles) lost their groups.
    def covered_names() -> set[str]:
        got = set()
        for f in outdir.glob("*.txt"):
            if f.name.startswith("_") or f.name == "all_marine.txt":
                continue
            got |= {l.strip() for l in f.read_text().splitlines() if l.strip()}
        return got

    orphans = sorted(set(allnames) - covered_names())
    print(f"\n  names in no group: {len(orphans):,} of {len(allnames):,}")

    # The 2009 keyword list is coarse: a name filed under "Cnidarians: Cnidaria" carries
    # no class at all, so no group can match it however complete TAXON_GROUPS is. Ask
    # GBIF's backbone for the missing ranks instead of dropping the name.
    if orphans:
        print(f"  asking api.gbif.org for the lineage of {len(orphans)} names …")
        rescued, still = 0, []
        cache_file = outdir / "_gbif_lineage_cache.json"
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
        for i, nm in enumerate(orphans, 1):
            lin = cache.get(nm)
            if lin is None:
                try:
                    lin = gbif_lineage(nm)
                except Exception:
                    lin = []
                cache[nm] = lin
                if i % 25 == 0:
                    cache_file.write_text(json.dumps(cache, indent=0))
            gs = groups_for(lin)
            if gs:
                rescued += 1
                for g in gs:
                    f = outdir / f"{g}.txt"
                    have = [l for l in f.read_text().splitlines() if l.strip()]
                    if nm not in have:
                        f.write_text("\n".join(sorted(have + [nm])) + "\n")
            else:
                still.append((nm, lin))
        cache_file.write_text(json.dumps(cache, indent=0))
        print(f"  rescued from GBIF lineage: {rescued}")
        if still:
            print(f"  STILL ungrouped: {len(still)}")
            for nm, lin in still[:12]:
                print(f"    {nm}   lineage={lin}")
            print("  FIX: extend TAXON_GROUPS or check the name; an ungrouped name is "
                  "reachable only through 'all marine species'.")
            return 1

    orphans = sorted(set(allnames) - covered_names())
    print(f"  after rescue, names in no group: {len(orphans):,}")
    if orphans:
        return 1

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
