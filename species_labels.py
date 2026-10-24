"""Read species labels out of photo filenames.

A labelled photo carries its species as the filename PREFIX; anything after it is a
free-form tag. Two photos of one species cannot share a filename, so the tag is how a
second example is expressed:

    Nembrotha_chamberlaini.jpg          -> Nembrotha chamberlaini
    Nembrotha_chamberlaini_01.jpg       -> Nembrotha chamberlaini   (second example)
    Nembrotha_chamberlaini-2.jpg        -> Nembrotha chamberlaini
    Nembrotha_chamberlaini__c42.jpg     -> Nembrotha chamberlaini   (tag after "__")
    Nembrotha chamberlaini copy.jpg     -> Nembrotha chamberlaini
    Goniobranchus_sp1.jpg               -> Goniobranchus sp1       (provisional, distinct)
    Goniobranchus_sp 2.jpg              -> Goniobranchus sp2
    manta_birostris.jpg                 -> Manta birostris         (lowercase normalised)

Rejected on purpose, never guessed:
    OI3A2349.jpg, DSC_0123.jpg, P1012345.jpg      camera filenames
    c05__Philippines__001.jpg                     an untouched shortlist pick

The shape that does the rejecting: a genus-like first token, then a lowercase epithet
(or a provisional marker). That admits "manta birostris" while refusing "DSC 0123" and
"Lembeh Manta" — the last one matters because a location written BEFORE the species
would otherwise become a species called "Lembeh Manta".
"""
from __future__ import annotations

import pathlib
import re

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

GENUS = re.compile(r"^[A-Za-z][a-z\-]{2,}$")
EPITHET = re.compile(r"^(?:[a-z][a-z\-]{1,}|(?:sp|cf|aff|nr)[0-9]*)$")
PROVISIONAL = re.compile(r"(?:sp|cf|aff|nr)[0-9]*")
TRAILING_NUM = re.compile(r"[-_ ]?[0-9]+$")


def _tidy(tokens: list[str]) -> list[str]:
    """Drop a duplicate-counter suffix from the epithet: 'chamberlaini-2' is a name.

    Deliberately NOT applied to provisional markers: 'sp1' and 'sp10' are different
    provisional taxa, and stripping the digits collapsed them into one class.
    """
    if len(tokens) >= 2:
        tokens = list(tokens)
        if not PROVISIONAL.fullmatch(tokens[1]):
            tokens[1] = TRAILING_NUM.sub("", tokens[1]) or tokens[1]
    return tokens


def species_from_name(filename: str) -> str | None:
    """Filename -> 'Genus species', or None when it is not a label.

    Takes everything before a double underscore, else the underscore/space separated
    tokens, and requires a binomial (or genus + provisional marker) shape.
    """
    stem = pathlib.PurePath(filename).stem
    head = stem.split("__")[0] if "__" in stem else stem
    tokens = [t for t in re.split(r"[ _]+", head) if t]
    tokens = _tidy(tokens)
    if len(tokens) < 2:
        return None
    genus, epithet = tokens[0], tokens[1]
    # "Chelidonichthys sp 1" must stay distinct from "sp 2"
    if len(tokens) >= 3 and epithet in {"sp", "cf", "aff", "nr"} and tokens[2].isdigit():
        epithet = epithet + tokens[2]
    if not (GENUS.match(genus) and EPITHET.match(epithet)):
        return None
    return f"{genus[:1].upper()}{genus[1:].lower()} {epithet.lower()}"


def load_labels(folder: str | pathlib.Path) -> tuple[dict[str, list[pathlib.Path]],
                                                     list[pathlib.Path]]:
    """Read a folder of labelled photos.

    Returns (labels, unlabelled) where labels maps species -> [paths]. Anything that does
    not parse is returned separately rather than dropped: a silently ignored label is
    worse than a reported one.
    """
    folder = pathlib.Path(folder).expanduser()
    labels: dict[str, list[pathlib.Path]] = {}
    unlabelled: list[pathlib.Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in PHOTO_EXTS:
            continue
        sp = species_from_name(p.name)
        if sp is None:
            unlabelled.append(p)
        else:
            labels.setdefault(sp, []).append(p)
    return labels, unlabelled
