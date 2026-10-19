#!/usr/bin/env python3
"""Pure-logic tests for the GBIF candidate source: naming rules and group mapping.

No network, no models, no Qt — these run anywhere, which is why they are what CI runs.
They pin the parsing rules that were each written in response to a real wrong answer.
"""
import pathlib
import inspect
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gbif_candidates as GC

FAILS = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


print("\n1. authorship is stripped, so a name is usable as a candidate")
for raw, want in [("Chromodoris annae Bergh, 1877", "Chromodoris annae"),
                  ("Nembrotha chamberlaini Gosliner & Behrens, 1997",
                   "Nembrotha chamberlaini"),
                  ("Goniobranchus geometricus (Risbec, 1928)",
                   "Goniobranchus geometricus")]:
    got = GC.strip_authority(raw)
    check(f"{raw[:34]:<36} -> {got}", got == want, f"want {want}")

print("\n2. genus-rank facet entries are rejected, not turned into fake species")
# A QGroupBox-worthy lesson: these arrive as "Genus Author" and are two words.
for fake in ["Chromodoris Alder", "Doto Oken", "Tenellia A.Costa",
             "Noumea Risbec", "Aegires Lovén", "Chromodoris sp."]:
    check(f"reject {fake!r}", not GC.is_species_name(fake))

print("\n3. a lower-case author particle is not an epithet")
# "Dermatobranchus van Hasselt, 1824" is a GENUS-level record whose author starts
# with "van"; stripping to two tokens produced the fake "Dermatobranchus van".
for fake in ["Dermatobranchus van", "Chromodoris de", "Genus di"]:
    check(f"reject {fake!r}", not GC.is_species_name(fake))

print("\n4. genuine short epithets are NOT over-rejected")
# These are real and were the reason the rule above is a particle list, not a
# minimum-length rule.
for real in ["Hypselodoris roo", "Goniobranchus coi", "Hypselodoris iba",
             "Hallaxa iju", "Noumeaella isa", "Chromodoris annae"]:
    check(f"accept {real!r}", GC.is_species_name(real))

print("\n5. there is no depth filter, deliberately")
# Depth is reported inconsistently by GBIF, so any depth filter silently removes real
# species. Measured at Anilao, 100 km: 143 of 541 nudibranch species carry no depth at
# all, and 880 of 1,672 crustacean species are recorded only below 40 m. A control that
# can only subtract is a trap, so it was removed rather than given better presets.
check("no depth presets exist", not hasattr(GC, "DEPTH_BANDS"))
check("fetch_candidates still accepts a depth range for scripts",
      "depth" in inspect.signature(GC.fetch_candidates).parameters)
check("the depth parameter is documented as subtractive",
      "CARRY a depth reading" in (inspect.getdoc(GC.fetch_candidates) or "").replace("\n", " "),
      (inspect.getdoc(GC.fetch_candidates) or "")[:80])

print("\n6. group mapping: iNaturalist specs -> GBIF group names")
cases = [({"taxa": [47113]}, "nudibranchs"),
         ({"iconic": "Aves"}, "birds"),
         ({"taxa": [47273]}, "sharks_rays"),
         ({"taxa": [47533]}, "corals_anemones"),
         ({}, None),
         (None, None)]
for spec, want in cases:
    got = GC.group_from_spec(spec)
    check(f"{str(spec):<28} -> {got}", got == want, f"want {want}")

print("\n7. bony fishes have no GBIF key, and that is deliberate")
# /species/204 IS Actinopterygii in the backbone, but every fish occurrence record
# carries classKey=None, so taxonKey=204 returns 0 records. A key that yields a
# confident empty list is worse than no key.
name, rank = GC.GROUP_TAXA["bony_fishes"]
check("bony_fishes has no resolvable taxon", name is None, str((name, rank)))

print("\n8. radius becomes a bounding box, because facets are illegal with geoDistance")
lat_r, lon_r = GC.bbox_from_point(13.75, 120.93, 30)
la0, la1 = (float(x) for x in lat_r.split(","))
lo0, lo1 = (float(x) for x in lon_r.split(","))
check("lat range brackets the point", la0 < 13.75 < la1, lat_r)
check("lon range brackets the point", lo0 < 120.93 < lo1, lon_r)
check("0.27 deg of latitude for 30 km", abs((la1 - la0) - 0.5389) < 0.01, f"{la1-la0:.4f}")
check("longitude widened to keep the radius", (lo1 - lo0) > (la1 - la0))

print("\n9. a group may need SEVERAL taxa, and each entry may carry its own rank")
# Crustacea has no usable key in the backbone: species/match?name=Crustacea matches a
# GENUS (10996236) and Isopoda answers HIGHERRANK -> Arthropoda. The group was mapped to
# Decapoda alone, which restricted it to crabs/shrimps/lobsters — at Anilao, 0-40 m that
# returned 159 species where the union returns 220, losing mantis shrimps, krill,
# barnacles and isopods.
def taxa_of(group):
    """Group -> [(name, rank)] however the table expresses it."""
    names, shared = GC.GROUP_TAXA[group]
    if names is None:
        return [], shared
    if not isinstance(names, (list, tuple)):
        return [(names, shared)], shared
    out = []
    for item in names:
        if isinstance(item, (list, tuple)):
            out.append((item[0], item[1] if len(item) > 1 else shared))
        else:
            out.append((item, shared))
    return out, shared

crus, _ = taxa_of("crustaceans")
crus_names = [n for n, _ in crus]
check("crustaceans maps to several taxa", len(crus) >= 5, str(crus_names))
check("crustaceans is not Decapoda-only", "Decapoda" not in crus_names, str(crus_names))
check("crustacean entries carry ranks", all(r for _, r in crus), str(crus))
check("Malacostraca included", "Malacostraca" in crus_names, str(crus_names))
check("barnacle class included (Maxillopoda)", "Maxillopoda" in crus_names, str(crus_names))
check("copepods included", "Copepoda" in crus_names, str(crus_names))

print("\n10. no group may use a name that resolves to a GENUS")
# "Crustacea" and "Tunicata" are real genera (10996236, 10971194) AND the obvious group
# words. Using either as a filter would silently restrict a group to one genus.
check("GENUS is rejected as a group rank", "GENUS" in GC.BAD_RANKS, str(GC.BAD_RANKS))
check("SPECIES is rejected as a group rank", "SPECIES" in GC.BAD_RANKS, str(GC.BAD_RANKS))
all_names = []
for g in GC.GROUP_TAXA:
    all_names.extend(n for n, _ in taxa_of(g)[0])
check("'Crustacea' never used as a taxon", "Crustacea" not in all_names, "Crustacea")
check("'Tunicata' never used as a taxon", "Tunicata" not in all_names, "Tunicata")

print("\n11. broad beats narrow: the group must not exclude the obvious cases")
# Each of these was previously a single narrow taxon that silently dropped members.
expect = {
    "echinoderms":    ["Echinodermata"],
    "worms":          ["Annelida", "Nemertea", "Sipuncula", "Chaetognatha"],
    "jellyfish":      ["Scyphozoa", "Cubozoa", "Ctenophora"],
    "tunicates":      ["Ascidiacea", "Thaliacea", "Appendicularia"],
    "algae_seaweed":  ["Rhodophyta", "Ochrophyta", "Chlorophyta"],
    "sharks_rays":    ["Elasmobranchii", "Holocephali"],
    "sea_slugs":      ["Nudibranchia", "Anaspidea"],
    "dugongs_manatees": ["Sirenia"],
}
for g, must in expect.items():
    names = [n for n, _ in taxa_of(g)[0]]
    check(f"{g} includes {', '.join(must)}", all(m in names for m in must), str(names))

# the specific narrow mappings that caused the bug must never come back
check("echinoderms is not Asteroidea-only",
      "Asteroidea" not in [n for n, _ in taxa_of("echinoderms")[0]], "Asteroidea")
check("dugongs_manatees is not Dugongidae-only",
      "Dugongidae" not in [n for n, _ in taxa_of("dugongs_manatees")[0]], "Dugongidae")
check("worms is not Annelida-only",
      len(taxa_of("worms")[0]) > 1, str(taxa_of("worms")[0]))

print()
if FAILS:
    print(f"GBIF NAMING RULES: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("GBIF NAMING RULES: ALL PASS")
