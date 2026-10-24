"""Group coverage: every marine name must land in at least one group.

This is the test for the failure that actually happened. Groups are assigned by matching
taxon names in a species' lineage, so any class missing from TAXON_GROUPS silently strands
its whole contents in "all marine species" — 639 of 7,405 names were unreachable that way
(cone shells, cowries, egg cowries, clams, chitons, brittle stars, comb jellies,
crocodiles). No CI test noticed, because nothing asked the question.

No network here: gbif_lineage() is the only part that talks to GBIF, and the build uses it
for the handful of names whose 2009 lineage has no class in it.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import build_mkw_candidates as B

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  ok   {name}")
    else:
        fails += 1
        print(f"  FAIL {name}  {detail}")


print("1. the classes that were missing now match")
cases = {
    "spider crab (Decapoda)": (["Animalia", "Arthropoda", "Malacostraca", "Decapoda",
                               "Epialtidae", "Xenocarcinus"], {"crabs_shrimps",
                                                               "crustaceans"}),
    "cone shell (Gastropoda)": (["Animalia", "Mollusca", "Gastropoda", "Conidae", "Conus"],
                                {"gastropods"}),
    "sea hare (Gastropoda/Anaspidea)": (["Animalia", "Mollusca", "Gastropoda",
                                         "Anaspidea", "Aplysia"],
                                        {"gastropods", "sea_slugs"}),
    "clam (Bivalvia)": (["Animalia", "Mollusca", "Bivalvia", "Veneridae"], {"bivalves"}),
    "chiton (Polyplacophora)": (["Animalia", "Mollusca", "Polyplacophora"], {"bivalves"}),
    "brittle star (Ophiuroidea)": (["Animalia", "Echinodermata", "Ophiuroidea"],
                                   {"echinoderms"}),
    "comb jelly (Ctenophora)": (["Animalia", "Ctenophora"], {"jellyfish"}),
    "crocodile (Crocodylia)": (["Animalia", "Chordata", "Reptilia", "Crocodylia"],
                               {"crocodiles"}),
    "marine iguana (Iguanidae)": (["Animalia", "Chordata", "Squamata", "Iguanidae"],
                                  {"crocodiles"}),
    "sea spider (Pycnogonida)": (["Animalia", "Arthropoda", "Pycnogonida"],
                                 {"sea_spiders"}),
    "salp (Thaliacea)": (["Animalia", "Chordata", "Thaliacea"], {"tunicates"}),
    "sea otter (Mustelidae)": (["Animalia", "Chordata", "Mammalia", "Carnivora",
                                "Mustelidae"], {"seals"}),
}
for label, (lineage, want) in cases.items():
    got = set(B.groups_for(lineage))
    check(label, want <= got, f"want {want}, got {got}")

print("2. a nudibranch is deliberately in several groups")
got = set(B.groups_for(["Animalia", "Mollusca", "Gastropoda", "Nudibranchia",
                        "Chromodoris"]))
check("nudibranch", {"nudibranchs", "sea_slugs", "gastropods"} <= got, got)

print("3. matching is on whole words, not substrings")
# "_" is a word character, so a made-up name must not satisfy "Gastropoda"
check("Gastropoda_x does not match",
      "gastropods" not in B.groups_for(["Animalia", "Mollusca", "Gastropoda_x"]),
      B.groups_for(["Animalia", "Mollusca", "Gastropoda_x"]))
check("case-insensitive",
      "gastropods" in B.groups_for(["animalia", "mollusca", "gastropoda"]))
check("Gastropoda matches", "gastropods" in B.groups_for(["Gastropoda"]))

print("4. a coarse lineage is honestly reported as ungrouped, not guessed")
# The 2009 list files some names under a phylum only; the build must see that as a gap
# and go and ask GBIF, not quietly put a coral in the gastropods.
check("phylum-only lineage yields no group", B.groups_for(["Animalia", "Mollusca"]) == [],
      B.groups_for(["Animalia", "Mollusca"]))
check("empty lineage yields no group", B.groups_for([]) == [])
check("none yields no group", B.groups_for(None) == [])

print("5. no taxon is listed under two contradictory groups by accident")
for taxon, want in (("Ophiuroidea", "echinoderms"), ("Gastropoda", "gastropods"),
                    ("Pycnogonida", "sea_spiders"), ("Ctenophora", "jellyfish"),
                    ("Crocodylia", "crocodiles")):
    check(f"{taxon} -> {want}", want in B.groups_for([taxon]), B.groups_for([taxon]))

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
