"""Species you add to the marine list yourself, kept where regeneration cannot erase them.

The group files (`nudibranchs.txt`, `sea_slugs.txt`, …) are GENERATED from a MarineLife
Keywords export by `make_marine_lists.py`. Appending to one of those looks like it works and
then quietly disappears the next time the lists are rebuilt — an addition that cannot survive
regeneration is not a saved addition, it is a note to oneself.

So additions live in `_user_added.txt` beside them, which the generator never writes, and which
every marine-list load unions in. Names are matched case-insensitively and appended once.

Pure standard library: no Qt, no numpy, so the rules can be tested without a GUI.
"""

from __future__ import annotations

import re
from pathlib import Path

ADDITIONS_NAME = "_user_added.txt"

HEADER = """# Marine species you added yourself.
#
# One scientific name per line — "Genus species". Blank lines and lines starting with # are
# ignored. These names are unioned into every marine list this app loads, and they survive
# regenerating the group files (make_marine_lists.py never touches this file).
#
# Delete a line here to stop a species being offered; nothing else refers to it.
"""

# Two words, letters only apart from the usual taxonomic furniture: spaces, hyphen, apostrophe.
# Deliberately strict — this file feeds a classifier, and a camera filename or a sentence
# entered here would be encoded as if it were a species.
_GENUS_RE = re.compile(r"^[A-ZÀ-Þ][\w'’\-]{2,}$", re.UNICODE)
_EPITHET_RE = re.compile(r"^[a-zà-þ][\w'’\-]*$", re.UNICODE)

# Rank qualifiers are not species. A list containing "Chromodoris sp." invites a
# match that names nothing.
_QUALIFIERS = {"sp", "sp.", "spp", "spp.", "cf", "cf.", "aff", "aff.", "nr", "nr."}

_PLACEHOLDERS = {"?", "??", "unknown", "unidentified", "n/a", "none", "null", "-"}


def additions_path(marine_dir: str | Path) -> Path:
    """Where the additions file lives, given the folder holding the group lists."""
    return Path(marine_dir) / ADDITIONS_NAME


def clean(name: str) -> str:
    """One canonical spelling for a name: collapsed whitespace, no trailing punctuation."""
    return " ".join(str(name or "").split()).strip(" ,;")


def reject_reason(name: str) -> str | None:
    """Why this is not a usable species name, or None if it is.

    A wrong name here is not cosmetic: it becomes a candidate the classifier can rank, so a
    filename like `OI3A2349` or a stray word like "nudibranch" would be scored as if it named a
    species. Better to refuse with a reason than to accept and pollute.
    """
    n = clean(name)
    if not n:
        return "the name is empty"
    if n.lower() in _PLACEHOLDERS:
        return "that is a placeholder, not a species"
    words = n.split()
    if len(words) < 2:
        return 'a species name needs two words — "Genus species"'
    if len(words) > 3:
        return "that is more than a species name (genus, epithet, at most a subspecies)"
    if not _GENUS_RE.match(words[0]):
        return "the genus must start with a capital letter"
    if words[1].lower() in _QUALIFIERS:
        return "a rank qualifier (sp., cf.) is not a species"
    for w in words[1:]:
        if not _EPITHET_RE.match(w):
            return "the epithet should be lowercase, like “poindimiei”"
    return None
    return None


def load(path: str | Path) -> list[str]:
    """The names in the additions file, in file order, without blanks or comments."""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[str] = []
    seen: set[str] = set()
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n = clean(line)
        k = n.lower()
        if n and k not in seen:
            seen.add(k)
            out.append(n)
    return out


def add(path: str | Path, name: str) -> dict:
    """Append one name. Returns {added, reason, total, name}.

    Idempotent: adding the same species twice is refused with a reason rather than
    duplicated, because a duplicate in a candidate list is encoded twice for nothing.
    """
    n = clean(name)
    if not n:
        return {"added": False, "reason": "the name is empty", "name": n,
                "total": len(load(path))}

    # Duplicate first, validation second: "chromodoris ANNAE" is the species already in the
    # file, and telling the user "the genus must start with a capital" would be true but
    # unhelpful — what they need to know is that it is already there.
    existing = load(path)
    if any(e.lower() == n.lower() for e in existing):
        return {"added": False, "reason": f"{n} is already in the file",
                "name": n, "total": len(existing)}

    why = reject_reason(n)
    if why:
        return {"added": False, "reason": why, "name": n, "total": len(existing)}

    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        new_file = not p.is_file()
        with p.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write(HEADER)
            fh.write(n + "\n")
    except OSError as exc:
        return {"added": False, "reason": f"could not write {p.name}: {exc}",
                "name": n, "total": len(existing)}
    return {"added": True, "reason": "", "name": n, "total": len(existing) + 1}


def membership(marine_dir: str | Path, name: str) -> dict:
    """Is this species already in the marine lists? Returns {in_additions, in_groups}.

    Checked before adding anything, because a list that already contains the species does not
    need an addition — and an additions file full of names the generated lists already hold is
    noise that hides the ones that actually matter.
    """
    n = clean(name)
    key = n.lower()
    out: dict = {"in_additions": False, "in_groups": []}
    if not key:
        return out
    d = Path(marine_dir)
    add_file = additions_path(d)
    if any(x.lower() == key for x in load(add_file)):
        out["in_additions"] = True
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.txt")):
        if f.name == ADDITIONS_NAME:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and clean(line).lower() == key:
                out["in_groups"].append(f.stem)
                break
    return out


def merge(base: list[str], extra: list[str]) -> list[str]:
    """`base` unchanged and in its own order, then any of `extra` it does not already hold.

    Order is information here — GBIF returns species most-recorded first, and the marine lists
    are alphabetical — so additions go on the end rather than being folded in, which keeps the
    original ranking readable and makes the additions easy to spot.
    """
    seen = {clean(b).lower() for b in base if clean(b)}
    out = [clean(b) for b in base if clean(b)]
    for e in extra:
        n = clean(e)
        k = n.lower()
        if n and k not in seen:
            seen.add(k)
            out.append(n)
    return out
