#!/usr/bin/env python3
"""The filename -> species rule: every naming style a person will actually produce.

This ran only in a scratch directory once, was pruned, and had to be rewritten — so it
lives in the repo now. The rule decides what a trained classifier can learn from, so it
is worth pinning: a parser that silently drops a label, or silently merges two species
into one class, corrupts the training set in ways that look like model failure later.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from species_labels import species_from_name

CASES = [
    # plain binomials and the ways people express a second photo of the same species
    ("Nembrotha_chamberlaini.jpg", "Nembrotha chamberlaini"),
    ("Nembrotha_chamberlaini_01.jpg", "Nembrotha chamberlaini"),
    ("Nembrotha_chamberlaini-2.jpg", "Nembrotha chamberlaini"),
    ("Nembrotha_chamberlaini__b.jpg", "Nembrotha chamberlaini"),
    ("Nembrotha_chamberlaini__c42.jpg", "Nembrotha chamberlaini"),
    ("Nembrotha chamberlaini copy.jpg", "Nembrotha chamberlaini"),
    ("manta_birostris_1.jpg", "Manta birostris"),
    ("manta_birostris_2.jpg", "Manta birostris"),
    ("manta_birostris_10.jpg", "Manta birostris"),
    # case: a lowercase genus is normalised, so it merges with the capitalised spelling
    ("manta_birostris.jpg", "Manta birostris"),
    ("Manta_birostris.jpg", "Manta birostris"),
    # provisional taxa stay distinct from each other
    ("Goniobranchus_sp1.jpg", "Goniobranchus sp1"),
    ("Goniobranchus_sp10__Lembeh.jpg", "Goniobranchus sp10"),
    ("Chelidonichthys_sp_1.jpg", "Chelidonichthys sp1"),
    ("Chelidonichthys_sp 2.jpg", "Chelidonichthys sp2"),
    # extra words after the binomial are tags, ignored
    ("Chromodoris_annae_subsp_orientalis.jpg", "Chromodoris annae"),
    ("Pteraeolidia_ianthina_Lembeh_2019.jpg", "Pteraeolidia ianthina"),
    ("Manta_birostris__Lembeh.jpg", "Manta birostris"),
    ("Manta_birostris_Lembeh_1.jpg", "Manta birostris"),
    # NOT labels: camera filenames and untouched shortlist picks
    ("OI3A2349.jpg", None),
    ("DSC_0123.jpg", None),
    ("P1012345.jpg", None),
    ("c05__Philippines__001.jpg", None),
    # a location BEFORE the species must not become a species name; it parses as
    # unlabelled (capitalised epithet fails the binomial test) and is reported as such
    ("Lembeh_Manta_birostris.jpg", None),
    # two tokens that are not a binomial
    ("Anilao_2024.jpg", None),
]

fails = 0
for name, want in CASES:
    got = species_from_name(name)
    if got != want:
        fails += 1
        print(f"  [FAIL] {name:<42} got {got!r} want {want!r}")

print(f"\nNAMING RULE: {len(CASES) - fails}/{len(CASES)} naming styles parsed as intended")
if fails:
    print(f"NAMING RULE: {fails} FAILED")
    sys.exit(1)
print("NAMING RULE: ALL PASS")
