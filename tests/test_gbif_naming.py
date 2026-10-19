#!/usr/bin/env python3
"""Pure-logic tests for the GBIF candidate source: naming rules and group mapping.

No network, no models, no Qt — these run anywhere, which is why they are what CI runs.
They pin the parsing rules that were each written in response to a real wrong answer.
"""
import pathlib
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

print("\n5. depth bands are two, and 'land' is not offered as a filter")
check("exactly two depth options", len(GC.DEPTH_BANDS) == 2, str(list(GC.DEPTH_BANDS)))
check("one of them is unfiltered", any(v is None for v in GC.DEPTH_BANDS.values()))
check("the other requires a depth reading",
      any(v == "0,11000" for v in GC.DEPTH_BANDS.values()))
check("no 'land only' label (GBIF cannot filter terrestrial)",
      not any("land only" in k.lower() for k in GC.DEPTH_BANDS))

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

print("\n9. a group may need SEVERAL taxa, and crustaceans does")
# Crustacea has no usable key in the backbone: species/match?name=Crustacea matches a
# GENUS (10996236) and Isopoda answers HIGHERRANK -> Arthropoda. The group was mapped to
# Decapoda alone, which restricted it to crabs/shrimps/lobsters — at Anilao, 0-40 m that
# returned 159 species where the union returns 220, losing mantis shrimps, krill,
# barnacles and isopods. The mapping must stay a union of crustacean classes.
crus, cr_rank = GC.GROUP_TAXA["crustaceans"]
check("crustaceans maps to several taxa", isinstance(crus, (list, tuple)) and len(crus) >= 5,
      str(crus))
check("crustaceans is not Decapoda-only", "Decapoda" not in (crus if isinstance(crus, (list, tuple)) else [crus]),
      str(crus))
check("crustacean classes carry a rank hint", cr_rank == "CLASS", str(cr_rank))
check("Malacostraca included", "Malacostraca" in crus, str(crus))
check("barnacle class included (Maxillopoda)", "Maxillopoda" in crus, str(crus))
check("copepods included", "Copepoda" in crus, str(crus))
check("'Crustacea' is never used as the taxon name",
      "Crustacea" not in crus, str(crus))

print("\n10. echinoderms means the phylum, not starfish")
# Asteroidea is starfish only; the marine lists mean the phylum (urchins, cucumbers,
# brittle stars, feather stars).
echo_name, echo_rank = GC.GROUP_TAXA["echinoderms"]
check("echinoderms maps to the phylum", echo_name == "Echinodermata", str(echo_name))
check("echinoderms uses a phylum rank hint", echo_rank == "PHYLUM", str(echo_rank))
check("echinoderms is not Asteroidea-only", echo_name != "Asteroidea", str(echo_name))

print()
if FAILS:
    print(f"GBIF NAMING RULES: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("GBIF NAMING RULES: ALL PASS")
