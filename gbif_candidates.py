"""GBIF candidate source for Species ID Studio.

Builds a *regional, ranked* candidate species list from GBIF occurrence records.
This is the first candidate source in the app that is location-aware: the others
are global lists (the 2009 MarineLife keyword lists) or iNaturalist queries.

Design notes, all verified against the live API (see references/keyword-list-ingestion.md):

* The regional query uses **decimalLatitude/decimalLongitude RANGES** to express a
  bounding box, NOT `geoDistance` and NOT `geometry`. That is not a preference:
  GBIF refuses to return facets for geometry/geoDistance queries, and without a
  facet there is no way to get a species list in one request.
* `facet=scientificName` returns species names directly. (The occurrence endpoint
  also has `speciesKey`, but that yields numeric keys needing one resolution call
  each.) Names come back with taxonomic authorship attached, which is stripped.
* There is **no marine filter**. `isMarine` and `habitat` are not parameters of the
  occurrence endpoint at all and are silently ignored. The practical marine filter
  is `depth`: only marine records carry it, so choosing a depth band implicitly
  restricts results to marine species.
* GBIF counts measure **survey effort, not abundance**. A species high in the list
  is one that has been catalogued often there, not necessarily the commonest.

No API key is needed.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.gbif.org/v1"
UA = {"User-Agent": "SpeciesIDStudio/1.0 (local offline research tool; "
                    "python-urllib)"}

CACHE = Path.home() / ".cache" / "species_studio" / "gbif_keys.json"

# Ranks that are never a valid answer for a GROUP filter. "Crustacea" and "Tunicata"
# both match real genera (10996236, 10971194) exactly, so EXACT is not enough.
BAD_RANKS = {"GENUS", "SPECIES", "SUBSPECIES", "VARIETY", "FORM", "SECTION"}

# Group -> the taxon name to resolve against GBIF. Kept deliberately in step with
# TAXON_GROUPS in build_mkw_candidates.py so the marine lists and the GBIF lists
# use the same vocabulary for the same group names.
GROUP_TAXA = {
    # DESIGN RULE: broad beats precise. A group that silently excludes the right
    # answer gives BioCLIP a ZERO chance of picking it, so for every group the widest
    # defensible taxon is used, and where no single key covers the group, several are
    # unioned. Extra noise costs a little ranking accuracy; a missing candidate costs
    # the entire identification. Verified against GBIF — every name below resolves
    # EXACT at the rank given, and several obvious broad names do NOT exist in the
    # backbone at all (Heterobranchia, Chondrichthyes, Actinopterygii, Sacoglossa,
    # Euthyneura, Teleostei, Osteichthyes), which is why the unions look odd.
    "nudibranchs": ("Nudibranchia", "ORDER"),
    # "sea slugs" in a keywords list is broader than Nudibranchia — it includes the sea
    # hares. Sacoglossa and Euthyneura have no exact match, so they cannot be added.
    "sea_slugs": ([("Nudibranchia", "ORDER"), ("Anaspidea", "ORDER"),
                   ("Aplysiomorpha", "ORDER")], None),
    # Crustacea is NOT usable: `species/match?name=Crustacea` returns an exact match on
    # a GENUS keyed 10996236, "Isopoda" answers HIGHERRANK -> Arthropoda, and crustacea
    # is paraphyletic so no single key covers it. Mapping this to Decapoda (as it was)
    # restricted the group to crabs/shrimps/lobsters: at Anilao, 0-40 m that returned
    # 159 species where the union returns 220. Keys read from GBIF by matching a real
    # species of each class (barnacles resolve to Maxillopoda, not Thecostraca).
    "crustaceans": ([("Malacostraca", "CLASS"), ("Maxillopoda", "CLASS"),
                     ("Copepoda", "CLASS"), ("Branchiopoda", "CLASS"),
                     ("Ostracoda", "CLASS"), ("Remipedia", "CLASS")], "CLASS"),
    "crabs_shrimps": ("Decapoda", "ORDER"),
    "cephalopods": ("Cephalopoda", "CLASS"),
    "corals_anemones": ("Anthozoa", "CLASS"),
    # Asteroidea is starfish only. "echinoderms" means the phylum: urchins, cucumbers,
    # brittle stars and feather stars too.
    "echinoderms": ("Echinodermata", "PHYLUM"),
    "sea_cucumbers": ("Holothuroidea", "CLASS"),
    # Elasmobranchii is sharks, rays and skates; Holocephali adds the chimaeras, so a
    # diver who photographs one does not get an empty list.
    "sharks_rays": ([("Elasmobranchii", "CLASS"), ("Holocephali", "CLASS")], "CLASS"),
    "bony_fishes": (None, None),
    "turtles": ("Testudines", "CLASS"),
    "cetaceans": ("Cetacea", "ORDER"),
    "seals": ("Pinnipedia", "FAMILY"),
    "sea_snakes": ("Hydrophiidae", "FAMILY"),
    # Dugongidae was dugongs only — manatees are Trichechidae. The order covers both.
    "dugongs_manatees": ("Sirenia", "ORDER"),
    "sponges": ("Porifera", "PHYLUM"),
    # Ascidiacea is sea squirts only; Thaliacea and Appendicularia are the pelagic
    # tunicates — salps and larvaceans — that a diver meets as jelly-like chains.
    "tunicates": ([("Ascidiacea", "CLASS"), ("Thaliacea", "CLASS"),
                   ("Appendicularia", "CLASS")], "CLASS"),
    "bryozoans": ("Bryozoa", "PHYLUM"),
    # "Worms" is not a taxon. Restricting to Annelida dropped ribbon worms, peanut
    # worms and arrow worms — all of them worm-shaped things a diver photographs.
    "worms": ([("Annelida", "PHYLUM"), ("Nemertea", "PHYLUM"),
               ("Sipuncula", "PHYLUM"), ("Chaetognatha", "PHYLUM")], "PHYLUM"),
    # Scyphozoa is true jellyfish only. Box jellies are Cubozoa, comb jellies a separate
    # phylum, and the hydrozoans include the siphonophores (Portuguese man o' war).
    "jellyfish": ([("Scyphozoa", "CLASS"), ("Cubozoa", "CLASS"),
                   ("Hydrozoa", "CLASS"), ("Ctenophora", "PHYLUM")], None),
    # Rhodophyta was red algae only, silently excluding kelp and sea lettuce.
    "algae_seaweed": ([("Rhodophyta", "PHYLUM"), ("Ochrophyta", "PHYLUM"),
                       ("Chlorophyta", "PHYLUM")], "PHYLUM"),
    # Land groups, so every choice in the app's group selector has a GBIF answer.
    "birds": ("Aves", "CLASS"),
    "insects": ("Insecta", "CLASS"),
    "spiders": ("Arachnida", "CLASS"),
    "mammals": ("Mammalia", "CLASS"),
    "reptiles": ("Squamata", "ORDER"),
    "amphibians": ("Amphibia", "CLASS"),   # needs the rank hint, see below
    "plants": ("Plantae", "KINGDOM"),
    "fungi": ("Fungi", "KINGDOM"),
    "snails_slugs": ("Gastropoda", "CLASS"),
    "clams": ("Bivalvia", "CLASS"),
}

# Why several of these names are NOT the obvious ones. GBIF's /species/match
# returns HIGHERRANK for a name it cannot place, meaning it substitutes an ANCESTOR
# and hands back that ancestor's key. Used blindly the filter silently widens:
#     "Sirenia"   -> HIGHERRANK, rank=CLASS, scientificName='Mammalia'
#     "Reptilia"  -> HIGHERRANK, rank=PHYLUM, scientificName='Chordata'
#     "Actinopterygii" + kingdom=Animalia -> HIGHERRANK, scientificName='Animalia'
# So "dugongs & manatees" would have returned every mammal, and "reptiles" every
# vertebrate. resolve_taxon_key() now REJECTS HIGHERRANK outright, and the names
# below were each verified to return matchType=EXACT:
#     Chondrichthyes -> nothing  => Elasmobranchii (121)
#     Chelonioidea   -> nothing  => Testudines (11418114)  [also includes tortoises]
#     Hydrophiinae   -> nothing  => Hydrophiidae (3542)
#     Sirenia        -> Mammalia => Dugongidae (9462)      [dugongs; not manatees]
#     Reptilia       -> Chordata => Squamata (11592253)    [not tuataras]
#     Amphibia       -> only with rank=CLASS => key 131
# A name that still cannot be resolved is deliberately left as None: the query then
# runs with no taxon filter rather than with a WRONG one. That is the case for
# bony fishes.
#
# BONY FISHES — the real reason, corrected. An earlier note here said
# "Actinopterygii is absent from GBIF's match index", which is only half true and
# hid the actual cause. Verified against the live API:
#   * /species/204 IS Actinopterygii, rank=CLASS, status=ACCEPTED, numDescendants
#     78,294, in the backbone dataset — the taxon page the user pointed at is right.
#   * But /occurrence/search?taxonKey=204 returns **0 records**, so using it would
#     make the app answer "no fish here" for every place on earth.
#   * Cause: GBIF's fish OCCURRENCE RECORDS have `class` and `classKey` = None. A
#     Solenostomus paradoxus record carries order Syngnathiformes, family
#     Solenostomidae, phylum Chordata — and class None. The control nudibranch record
#     (Nembrotha chamberlaini) does carry class Gastropoda / classKey 225. Ray-finned
#     fishes are simply not linked into the class node, so nothing is indexed there.
#   * Ranks below class DO work: Perciformes 587 -> 42.2M records, Cypriniformes 1153
#     -> 11.9M, Syngnathiformes 773 -> 576K. But `taxonKey` accepts only ONE key (no
#     unions), and Perciformes alone excludes seahorses and pipefish — exactly the
#     animals a diver is most likely photographing.
# So: no filter for bony fishes, and say so. A WRONG key that returns a confident
# empty list is far worse than an honest absent filter.

# The app's group selector is shared with the iNaturalist source, so it carries
# iNat specs ({"iconic": "Aves"} / {"taxa": [47113]}) rather than these names.
# Keyed by the spec's canonical form so the two sources stay in step.
INAT_SPEC_TO_GROUP = {
    ("iconic", ("Aves",)): "birds",
    ("iconic", ("Insecta",)): "insects",
    ("iconic", ("Arachnida",)): "spiders",
    ("iconic", ("Mammalia",)): "mammals",
    ("iconic", ("Reptilia",)): "reptiles",
    ("iconic", ("Amphibia",)): "amphibians",
    ("iconic", ("Plantae",)): "plants",
    ("iconic", ("Fungi",)): "fungi",
    ("taxa", (372234,)): "turtles",              # Chelonioidea
    ("taxa", (152871,)): "cetaceans",            # Cetacea
    ("taxa", (41736, 41687, 41764)): "seals",    # Otariidae/Phocidae/Odobenidae
    ("taxa", (492346,)): "sea_snakes",           # Hydrophiinae
    ("taxa", (46306,)): "dugongs_manatees",      # Sirenia
    ("taxa", (47178,)): "bony_fishes",           # Actinopterygii
    ("taxa", (47273,)): "sharks_rays",           # Chondrichthyes
    ("taxa", (47113,)): "nudibranchs",           # Nudibranchia
    ("taxa", (47187,)): "crabs_shrimps",         # Decapoda
    ("taxa", (47459,)): "cephalopods",           # Cephalopoda
    ("taxa", (47533,)): "corals_anemones",       # Anthozoa
    ("taxa", (47668, 47548)): "echinoderms",     # Asteroidea/Echinoidea
    ("taxa", (47720,)): "sea_cucumbers",         # Holothuroidea
    ("taxa", (47584,)): "clams",                 # Bivalvia
    ("taxa", (48824, 47811)): "sponges",         # Porifera/Ascidiacea
    ("taxa", (47114,)): "snails_slugs",          # Gastropoda
}


def spec_key(spec: dict | None) -> tuple | None:
    """Canonical, hashable form of an iNaturalist group spec."""
    if not spec:
        return None
    if "iconic" in spec:
        return ("iconic", (spec["iconic"],))
    if "taxa" in spec:
        return ("taxa", tuple(spec["taxa"]))
    return None


def group_from_spec(spec: dict | str | None) -> str | None:
    """Map the app's group selector (an iNat spec) to a GBIF group name.

    Returns None for "no group"/"not sure"/the megafauna union, meaning: run with
    no taxon filter. None is a legitimate answer, not a failure.
    """
    if isinstance(spec, str):
        return spec if spec in GROUP_TAXA else None
    k = spec_key(spec)
    return INAT_SPEC_TO_GROUP.get(k) if k else None


# Depth bands. Choosing one is also the only reliable way to say "marine only".
# Two options, not six — and the reason is measured, not aesthetic.
#
# At Anilao (nudibranchs, 100 km): "underwater, any depth" gives 399 species and is a
# strict SUPERSET of the 0-40 m band's 394 — every depth band was subtracting real
# animals. The 0-40 m band lost five mesophotic species (Halgerda scripta,
# H. mesophotica, H. profunda, H. takipsilim, Dermatobranchus leoni), so bands cost
# accuracy for no benefit.
#
# Critically, a depth band is NOT a marine filter. Depth is metadata that only some
# records carry: filtering on it drops species whose records happen to lack a depth
# reading, whether they are marine or not (records 14,849 -> 2,437, species 542 -> 399
# in the same box). Records without depth are a MIX of land species and marine species
# that simply never had a depth recorded.
#
# So the honest pair is "any" (most complete) and "underwater only" (records that
# carry a depth). "On land" cannot be selected for — GBIF has no isTerrestrial, exactly
# as it has no isMarine — and depth presence does not identify it either.
# No depth presets. The GUI no longer offers a depth control, because GBIF's depth
# field is populated inconsistently and a depth filter can only ever SUBTRACT: at
# Anilao, 143 of 541 nudibranch species carry no depth at all, so any depth filter hides
# them, and 880 of 1,672 crustacean species there are recorded only below 40 m.
# `fetch_candidates(depth=...)` remains available for scripts that know what they want.

DEFAULT_RADIUS_KM = 30.0

# The facet response is NOT capped by the API. Measured against worldwide
# Nudibranchia: facetLimit=5000 returned all 4,993 species in 269 KB / 0.5 s, and an
# all-taxa query returned 30,000 buckets in 1.67 MB / 8.3 s. An earlier version of
# this module clamped to 1000 on the assumption that 1000 was the ceiling — it is
# not, and that clamp silently truncated large regions to their top 1000 species.
FACET_LIMIT_MAX = 50_000

# Default is a judgement call, not a limit: at dive-site scale (a 15-50 km box) the
# whole list is well under it — Anilao at 25 km is 347 nudibranchs and 486 species
# with no group filter at all — while keeping the BioCLIP encode quick (~60 s for
# 1,000 names). Larger regions need a larger value, and fetch_candidates() reports
# when the result actually hit the ceiling so truncation is never silent.
DEFAULT_FACET_LIMIT = 1000


def _get(path: str, **params) -> dict:
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 400s are programmer error (bad parameter), not worth retrying.
            if e.code == 400:
                raise ValueError(
                    f"GBIF rejected the query: {e.read().decode()[:200]}") from e
            if attempt == 2:
                raise
        except Exception:
            if attempt == 2:
                raise
        time.sleep(1.5 * (attempt + 1))
    return {}


# --------------------------------------------------------------------------- keys

def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(d: dict) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(d, indent=1, sort_keys=True))
    except Exception:
        pass


def resolve_taxon_key(name: str, rank: str | None = None
                      ) -> tuple[int | None, str]:
    """GBIF backbone key for a taxon name. Cached on disk.

    Returns (key, label). The key is None when GBIF cannot place the name — and
    crucially also when it answers HIGHERRANK, which means it is offering an
    ANCESTOR's key instead ("Sirenia" -> Mammalia, "Reptilia" -> Chordata, both
    verified). Accepting those would silently widen a group filter to an entire
    class or phylum, so they are rejected here rather than used.

    An EXACT answer can still be the wrong KIND of taxon: "Crustacea" matches a GENUS
    keyed 10996236 and "Tunicata" a GENUS keyed 10971194 — real genera whose names
    collide with the group words. A group filter at genus rank would silently restrict
    a whole group to one genus, so genus and below are rejected too.
    """
    ck = f"{name}|{rank or ''}"
    cache = _load_cache()
    if ck in cache:
        e = cache[ck]
        if str(e.get("rank") or "").upper() in BAD_RANKS:
            return None, e.get("label", name)
        return e.get("key"), e.get("label", name)

    params = {"name": name, "strict": "false"}
    if rank:
        params["rank"] = rank
    d = _get("species/match", **params)
    key = d.get("usageKey") or d.get("speciesKey")
    mt = d.get("matchType")
    if mt != "EXACT" or not key:
        # NONE: unknown. HIGHERRANK: an ancestor, i.e. wrong. Either way, no filter.
        key = None
    elif str(d.get("rank") or "").upper() in BAD_RANKS:
        key = None
    resolved = d.get("scientificName") or name
    cache[ck] = {"key": key, "label": resolved, "matchType": mt,
                 "rank": d.get("rank")}
    _save_cache(cache)
    return key, resolved


# ------------------------------------------------------------------------ geometry

def bbox_from_point(lat: float, lon: float, radius_km: float
                    ) -> tuple[str, str]:
    """A lat/lon bounding box as two GBIF range strings.

    GBIF cannot facet a `geoDistance` (radius) query, so a radius is converted to a
    box. That is an approximation — it is a square, so corners are ~1.41x the
    radius away — and it is the reason this module takes a radius rather than
    promising a true circle.
    """
    dlat = radius_km / 111.32
    coslat = max(0.01, math.cos(math.radians(lat)))
    dlon = radius_km / (111.32 * coslat)
    latr = f"{max(-90.0, lat - dlat):.4f},{min(90.0, lat + dlat):.4f}"
    lonr = f"{max(-180.0, lon - dlon):.4f},{min(180.0, lon + dlon):.4f}"
    return latr, lonr


QUALIFIERS = {"sp", "sp.", "spp", "spp.", "cf", "cf.", "aff", "aff.", "indet",
              "indet.", "nr", "nr.", "species", "complex", "sensu", "s.l.", "s.s."}

# Author particles that start with a lower-case letter. "Dermatobranchus van Hasselt,
# 1824" is a GENUS-level record whose author begins with "van", so after stripping to
# two tokens it read as the fake species "Dermatobranchus van" — the lower-case-epithet
# test passes it because "van" is lower case. Compare "Dermatobranchus pustulosus van
# Hasselt, 1824", which strips correctly. Rejecting these particles is what separates
# a Dutch author from a specific epithet.
AUTHOR_PARTICLES = {"van", "von", "de", "der", "den", "del", "della", "di", "da",
                    "du", "la", "le", "dos", "das", "ter", "ten", "zu", "zur", "af",
                    "op", "ex", "et", "and", "in"}


def is_species_name(name: str) -> bool:
    """True only for a bare binomial, rejecting GBIF's genus-rank facet entries.

    The `scientificName` facet includes records identified only to genus, and those
    come back as "Genus Author" — e.g. "Chromodoris Alder", "Doto Oken",
    "Tenellia A.Costa". They are two words, so a word count cannot tell them apart
    from a real species, and strip_authority() happily turns the AUTHOR into an
    epithet. Nomenclature settles it: a species epithet is always lowercase
    (ICZN/ICN), while those genus-rank second tokens are capitalised author names.
    """
    toks = name.split()
    if len(toks) != 2:
        return False
    ep = toks[1]
    if not ep or not ep[0].islower():        # 'Alder', 'A.Costa', 'Lovén' -> genus rank
        return False
    if ep.lower().strip(".") in {q.strip(".") for q in QUALIFIERS}:
        return False
    if ep.lower() in AUTHOR_PARTICLES:       # 'van', 'de' -> an author, not an epithet
        return False
    if len(ep) < 2:
        return False
    if any(ch.isdigit() for ch in ep):
        return False
    return True


# -------------------------------------------------------------------------- names

def strip_authority(name: str) -> str:
    """'Chromodoris annae Bergh, 1877' -> 'Chromodoris annae'.

    The facet returns names with authorship, which must not reach a candidate list:
    an author string in a prompt biases the model toward that taxon.
    """
    s = name.strip()
    # Drop a trailing parenthesised authority: "Genus species (Risbec, 1928)"
    if "(" in s and s.rstrip().endswith(")"):
        s = s[:s.index("(")].strip()
    parts = s.split()
    out: list[str] = []
    for i, w in enumerate(parts):
        # Stop at the first token that is not part of a bare binomial.
        if i >= 2:
            break
        if any(ch.isdigit() for ch in w):
            break
        if "," in w or w.lower() in ("var.", "subsp.", "f.", "cf.", "sp."):
            break
        out.append(w)
    return " ".join(out) if len(out) >= 2 else s.split(",")[0].strip()


# --------------------------------------------------------------------------- main

def fetch_candidates(*, lat: float | None = None, lon: float | None = None,
                     radius_km: float = DEFAULT_RADIUS_KM,
                     bbox: tuple[float, float, float, float] | None = None,
                     group: str | None = None,
                     taxon_key: int | None = None,
                     depth: str | None = None,
                     limit: int = 1000,
                     min_count: int = 1) -> tuple[list[str], str]:
    """Return (names, human-readable provenance line).

    `group` is one of GROUP_TAXA. Either supply lat/lon (with radius) or an explicit
    bbox as (min_lat, max_lat, min_lon, max_lon).

    `depth` is a GBIF range string such as "0,40". It is NOT exposed in the GUI and
    should be used with care: it selects the records that happen to CARRY a depth
    reading, so it drops every record without one — 143 of 541 nudibranch species at
    Anilao — and every species recorded only outside the range.
    """
    facet_limit = max(1, min(int(limit), FACET_LIMIT_MAX))
    params: dict[str, object] = {"limit": 0, "facet": "scientificName",
                                 "facetLimit": facet_limit}
    if min_count > 1:
        params["facetMincount"] = min_count

    bits: list[str] = []

    # Geometry — must be the RANGE form, or facets are unavailable.
    if bbox:
        min_lat, max_lat, min_lon, max_lon = bbox
        params["decimalLatitude"] = f"{min_lat:.4f},{max_lat:.4f}"
        params["decimalLongitude"] = f"{min_lon:.4f},{max_lon:.4f}"
        bits.append(f"box {min_lat:.2f},{min_lon:.2f} to {max_lat:.2f},{max_lon:.2f}")
    elif lat is not None and lon is not None:
        latr, lonr = bbox_from_point(lat, lon, radius_km)
        params["decimalLatitude"] = latr
        params["decimalLongitude"] = lonr
        bits.append(f"~{radius_km:.0f} km around {lat:.3f},{lon:.3f}")
    else:
        raise ValueError("need either a bbox or lat/lon")

    # Taxonomy. `group` may be given either as a GBIF group name or as the app's
    # own selector value, which is an iNaturalist spec — accept both, because
    # passing an iNat spec straight through silently dropped the filter.
    label = "all taxa"
    unfiltered_group = None
    group_keys: list[tuple[int, str]] = []
    if taxon_key is None and group:
        # Branch explicitly rather than with a ternary: `group in GROUP_TAXA` is
        # evaluated before any isinstance guard inside a conditional expression,
        # and a dict is unhashable, so it raised TypeError for the GUI's spec dict.
        if isinstance(group, str):
            gname = group if group in GROUP_TAXA else group_from_spec(group)
        else:
            gname = group_from_spec(group)
        if gname and gname in GROUP_TAXA:
            tname, rank = GROUP_TAXA[gname]
            # A group may need SEVERAL taxa to cover it (Crustacea has no single key).
            # Union them; every one must still resolve EXACT or it is dropped.
            wanted = tname if isinstance(tname, (list, tuple)) else [tname]
            wanted = [w for w in wanted if w]
            if wanted:
                for one in wanted:
                    # An entry may carry its own rank, because a group's members can sit
                    # at different ranks: Scyphozoa is a CLASS, Ctenophora a PHYLUM, so
                    # "jellyfish" cannot share one hint.
                    if isinstance(one, (list, tuple)):
                        oname = one[0]
                        orank = one[1] if len(one) > 1 else rank
                    else:
                        oname, orank = one, rank
                    k, resolved = resolve_taxon_key(oname, orank)
                    if k:
                        group_keys.append((k, resolved))
                if group_keys:
                    taxon_key = group_keys[0][0]
                    label = (group_keys[0][1] if len(group_keys) == 1 else
                             f"{gname}: " + " + ".join(r for _, r in group_keys))
                else:
                    label = f"{gname} (no GBIF filter)"
            else:
                # Name the group the user actually chose. Reporting "all taxa" here
                # was misleading: they picked fishes, so say the filter is absent,
                # not that they asked for everything.
                label = f"{gname} (unfiltered)"
                unfiltered_group = gname
        # A group with no resolvable name deliberately runs unfiltered: a WRONG
        # ancestor filter is worse than none.
    elif taxon_key is not None:
        label = str(taxon_key)
    multi = len(group_keys) > 1
    if multi:
        # One query per key: `taxonKey` cannot take several values — repeated, comma-
        # separated and list forms all return 0, which is indistinguishable from
        # "nothing recorded here". So union separate queries instead.
        bits.append(f"taxon {label}")
    elif taxon_key is not None:
        params["taxonKey"] = taxon_key
        bits.append(f"taxon {label}")
    elif group:
        bits.append(f"no taxon filter — {label}")

    if depth:
        params["depth"] = depth
        bits.append(f"depth {depth.replace(',', '-')} m")
    # No month filter, deliberately. Measured at Anilao (nudibranchs, 15 km, 10-30 m):
    # 1,357 records / 314 species unfiltered, but only 2 records in January. GBIF
    # coverage is sparse and survey-driven per month, so a month filter measures when
    # people happened to dive there, not seasonality — and it fails silently, since a
    # 2-species answer reads as "not much here" rather than "filter too tight".

    requested = facet_limit
    counts: list[dict] = []
    total = 0
    capped = False
    if multi:
        for k, _r in group_keys:
            p = dict(params)
            p["taxonKey"] = k
            d = _get("occurrence/search", **p)
            total += d.get("count", 0)
            fs = (d.get("facets") or [{}])[0].get("counts", []) or []
            counts.extend(fs)
            if len(fs) >= requested:
                capped = True
    else:
        d = _get("occurrence/search", **params)
        facets = d.get("facets") or []
        counts = facets[0]["counts"] if facets else []
        total = d.get("count", 0)
        capped = len(counts) >= requested
    names, seen = [], set()
    for c in counts:
        n = strip_authority(c["name"])
        # Rejects the genus-rank facet entries ("Chromodoris Alder") that would
        # otherwise enter the candidate list as fake species.
        if not is_species_name(n) or n in seen:
            continue
        seen.add(n)
        names.append(n)

    # A group GBIF cannot filter (bony fishes) comes back as EVERYTHING recorded
    # there: measured 741 names, of which only 119 were fish — the rest birds, snakes,
    # crabs. Returning that when the user asked for "reef fish" is actively misleading,
    # so restrict to the group's own name list from marine_lists/ when one exists, and
    # say plainly that it was restricted.
    if unfiltered_group:
        wl = Path(__file__).parent / "marine_lists" / f"{unfiltered_group}.txt"
        if wl.is_file():
            allowed = {l.strip() for l in wl.read_text().splitlines() if l.strip()}
            before = len(names)
            names = [n for n in names if n in allowed]
            bits.append(f"restricted to your {len(allowed):,} {unfiltered_group} names "
                        f"(dropped {before - len(names)} that are not in it)")

    # total and capped are set by the query above: summed across keys when a group
    # needed several taxa, and capped if ANY of those queries hit the facet limit.
    info = (f"GBIF: {total:,} occurrence records | " + " | ".join(bits)
            + f" | {len(names)} species")
    if capped:
        info += (f"  NOTE: capped at {requested:,} — more species exist here, "
                 "raise the limit for the full list")
    return names, info


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="GBIF regional candidate list")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_KM)
    ap.add_argument("--group", choices=sorted(GROUP_TAXA))
    # Raw GBIF range, e.g. "0,40". Deliberately not a menu of presets: depth is
    # metadata that subtracts, and the CLI should make that explicit rather than
    # invite it. See the note above DEFAULT_RADIUS_KM.
    ap.add_argument("--depth", metavar="MIN,MAX",
                    help="GBIF depth range in metres; drops records without a depth "
                         "reading and species recorded only outside the range")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out")
    a = ap.parse_args()

    names, info = fetch_candidates(lat=a.lat, lon=a.lon, radius_km=a.radius,
                                  group=a.group,
                                  depth=a.depth,
                                  limit=a.limit)
    print(info)
    print(f"\n  {len(names)} candidates, most-recorded first:")
    for n in names[:15]:
        print(f"     {n}")
    if a.out:
        Path(a.out).write_text("\n".join(names) + "\n")
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
