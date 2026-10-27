"""Every group must map to a real GBIF taxon filter — including the ones GBIF has no class for.

The failure this guards: "bony_fishes" was (None, None), so the GBIF query ran with no taxon
filter, the caller intersected a top-1,000 species facet with the user's fish list, and that
facet is ordered by record count — at Anilao it is mostly land birds. "Reef & bony fishes"
therefore returned 119 species at Batangas and silently dropped every fish that did not make
the cut.

No network here: the structure is checked offline. Whether the names resolve is a live
question and belongs to the app's own provenance line.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gbif_candidates as G

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  ok   {name}")
    else:
        fails += 1
        print(f"  FAIL {name}  {detail}")


def entries(spec):
    """Normalise a GROUP_TAXA value to a list of (name, rank) pairs."""
    if spec is None:
        return []
    first = spec[0]
    if first is None:
        return []
    if isinstance(first, str):
        return [first]
    return list(first)


print("1. no group is silently unfiltered")
unfiltered = [k for k, v in G.GROUP_TAXA.items() if not entries(v)]
check("every group has at least one taxon", unfiltered == [], unfiltered)

print("2. bony fishes are filtered, and by fish-only taxa")
bf = entries(G.GROUP_TAXA["bony_fishes"])
check("bony_fishes is not (None, None)", G.GROUP_TAXA["bony_fishes"] != (None, None),
      G.GROUP_TAXA["bony_fishes"])
check("several orders are unioned", len(bf) >= 10, len(bf))
check("all at ORDER rank", all(r == "ORDER" for _, r in bf), set(r for _, r in bf))
names = [n for n, _ in bf]
check("includes Perciformes (GBIF keeps the newer orders inside it)",
      "Perciformes" in names, names[:5])
for must in ("Scorpaeniformes", "Tetraodontiformes", "Anguilliformes", "Lophiiformes"):
    check(f"includes {must}", must in names)
check("no duplicates", len(names) == len(set(names)), len(names) - len(set(names)))

print("3. groups that need a union still have one")
for group, least in (("crustaceans", 5), ("sharks_rays", 2), ("tunicates", 3),
                     ("worms", 4), ("jellyfish", 4), ("algae_seaweed", 3)):
    got = len(entries(G.GROUP_TAXA[group]))
    check(f"{group} unions {got} taxa (>= {least})", got >= least, got)

print("4. the ranks are ones the resolver accepts")
RANKS = {"KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES"}
bad = []
for group, spec in G.GROUP_TAXA.items():
    if isinstance(spec[0], str):
        if spec[1] not in RANKS:
            bad.append((group, spec[1]))
    else:
        for _, r in entries(spec):
            if r is not None and r not in RANKS:
                bad.append((group, r))
check("all ranks valid", bad == [], bad)

print("5. the app's fish group reaches the fish mapping")
# the group selector speaks iNaturalist taxon ids; 47178 is Actinopterygii there
check("iNat 47178 -> bony_fishes", G.group_from_spec({"taxa": [47178]}) == "bony_fishes",
      G.group_from_spec({"taxa": [47178]}))
check("a group name passes through", G.group_from_spec("bony_fishes") == "bony_fishes")
check("no group stays unfiltered", G.group_from_spec({}) is None)
check("unknown spec stays unfiltered", G.group_from_spec({"taxa": [999999999]}) is None)

print("6. an empty result says WHY it is empty instead of looking like missing data")
# The wrong-place failure, twice seen: "Bintang" resolves to the Papua highlands, where
# there are no nudibranchs — but 1,000 species of any kind ARE recorded, which is the
# diagnostic. Stub the API so this runs offline.
real_get = G._get
calls = []


def fake_get(path, **params):
    calls.append((path, dict(params)))
    if path == "species/match":
        return {"matchType": "EXACT", "rank": "ORDER", "usageKey": 980,
                "scientificName": "Nudibranchia"}
    if path == "occurrence/search":
        if "taxonKey" in params:
            return {"count": 0, "facets": [{"field": "SCIENTIFIC_NAME", "counts": []}]}
        return {"count": 12345,
                "facets": [{"field": "SCIENTIFIC_NAME", "counts": [
                    {"name": "Mellea ordinaria Linnaeus, 1758", "count": 900},
                    {"name": "Meekiaria Munroe, 1974", "count": 400}]}]}
    return {}


G._get = fake_get
try:
    names, info = G.fetch_candidates(lat=-4.70725, lon=140.43362, radius_km=100,
                                     group="nudibranchs")
finally:
    G._get = real_get
check("no names returned", names == [], names)
check("the info names the emptiness", "NOTHING in this group" in info, info[:200])
check("it says how much IS recorded", "2 species of any kind" in info, info[:260])
check("it quotes what is there, so land is obvious",
      "Mellea ordinaria" in info and "Meekiaria" in info, info[:260])
check("it blames the location, not the data", "wrong or inland" in info, info[:260])
check("it asked the unfiltered question exactly once",
      sum(1 for p, q in calls if p == "occurrence/search" and "taxonKey" not in q) == 1,
      [p for p, _ in calls])

# and when truly nothing is recorded there at all
def fake_get_empty(path, **params):
    if path == "species/match":
        return {"matchType": "EXACT", "rank": "ORDER", "usageKey": 980,
                "scientificName": "Nudibranchia"}
    return {"count": 0, "facets": [{"field": "SCIENTIFIC_NAME", "counts": []}]}


G._get = fake_get_empty
try:
    names, info = G.fetch_candidates(lat=0.0, lon=-160.0, radius_km=50,
                                     group="nudibranchs")
finally:
    G._get = real_get
check("an empty ocean point says nothing is recorded there",
      "nothing at all is recorded" in info, info[:220])

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
