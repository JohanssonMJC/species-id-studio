#!/usr/bin/env python3
"""
Lightroom keyword-list ingester.

A dive keyword list is a *hierarchy* (Marine Life > Crustaceans > Crabs >
Portunus...), and that hierarchy is the whole point: it is what lets you narrow
a list of thousands down to "this is a crustacean" without loading all of them,
and it costs nothing because it is already in the file — no API, no lookup,
works offline.

Lightroom exports keywords in a few shapes, so detect rather than assume:

  tab-indented outline        (Export Keywords -> .txt)
    Animals
    \tMarine Life
    \t\tCrustaceans
    \t\t\tCrabs

  separator-delimited flat    (keyword path separator: > | < » or /)
    Animals > Marine Life > Crustaceans > Crabs

  space-indented outline      (pasted out of the Keywords panel)

Public API:
    parse(text)            -> tree node
    walk(node)             -> yield (path_tuple, node)
    leaves(node)           -> list of path tuples (species-level entries)
    names_under(node, pat) -> leaf names whose path contains a node matching pat
    outline(node, depth)   -> printable hierarchy summary
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SEPARATORS = ("\t", " > ", ">", " | ", "|", " < ", "<", "»", "/")

# Nodes that are categories, not species — these must never become candidates,
# or BioCLIP will score "Crustaceans" as if it were a taxon.
CATEGORY_HINTS = {
    "animals", "animal", "marine life", "marinelife", "underwater", "sea life",
    "wildlife", "nature", "invertebrates", "invertebrate", "vertebrates",
    "fish", "fishes", "nudibranchs", "nudibranch", "crustaceans", "crustacean",
    "crabs", "shrimps", "lobsters", "molluscs", "mollusks", "cephalopods",
    "octopus", "squid", "corals", "anemones", "echinoderms", "starfish",
    "urchins", "sea cucumbers", "sponges", "tunicates", "worms", "tunicata",
    "gastropods", "bivalves", "sharks", "rays", "sea slugs", "flatworms",
    "places", "locations", "dive sites", "unknown", "unidentified", "misc",
    "other", "miscellaneous", "unassigned",
}

_RANK_WORDS = re.compile(
    r"^(class|order|family|suborder|superfamily|genus|subgenus|phylum|kingdom|"
    r"infraorder|subclass|section|tribe|clade)\b", re.I)


@dataclass
class Node:
    name: str
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None

    def path(self) -> tuple[str, ...]:
        parts, n = [], self
        while n is not None and n.name:
            parts.append(n.name)
            n = n.parent
        return tuple(reversed(parts))

    def add(self, parts: list[str]) -> None:
        cur = self
        for p in parts:
            p = p.strip()
            if not p:
                continue
            nxt = next((c for c in cur.children if c.name == p), None)
            if nxt is None:
                nxt = Node(p, parent=cur)
                cur.children.append(nxt)
            cur = nxt

    def depth(self) -> int:
        d, n = 0, self
        while n.parent is not None:
            d += 1
            n = n.parent
        return d


def _detect_separator(lines: list[str]) -> str | None:
    """Pick the separator that consistently splits lines into >1 part."""
    best, best_score = None, 0.0
    for sep in (" > ", " >", ">", " | ", " |", "|", " < ", "<", "»"):
        hits = sum(1 for l in lines if sep in l)
        if hits == 0:
            continue
        score = hits / max(1, len(lines))
        if score > best_score and hits >= max(2, len(lines) * 0.25):
            best, best_score = sep, score
    return best


def _detect_indent(lines: list[str]) -> bool:
    """True when most non-blank lines start with a tab or are indented."""
    ne = [l for l in lines if l.strip()]
    if not ne:
        return False
    if sum(1 for l in ne if l.startswith("\t")) >= max(2, len(ne) * 0.3):
        return True
    return sum(1 for l in ne if l[:1] in (" ", "\t")) >= max(2, len(ne) * 0.3)


def parse(text: str) -> Node:
    """Parse a keyword list into a tree. Auto-detects the export shape.

    Line endings are normalised first: some Lightroom-era exports (and anything
    that travelled through 2000s Mac software) use bare CR or CRLF, so splitting
    on "\\n" alone silently yields ONE giant line.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = [l.rstrip() for l in normalised.split("\n")]
    lines = [l for l in raw if l.strip() and not l.strip().startswith("#")]
    root = Node("")

    if not lines:
        return root

    # 1. separator-delimited flat paths
    sep = _detect_separator(lines)
    if sep:
        for l in lines:
            root.add([p.strip() for p in l.split(sep)])
        return root

    # 2. indented outline — classic parent-stack build
    if _detect_indent(lines):
        indents = []
        for l in lines:
            ind = len(l) - len(l.lstrip("\t "))
            if ind > 0:
                indents.append(ind)
        unit = min(indents) if indents else 1
        stack: list[tuple[int, Node]] = []
        for l in lines:
            stripped = l.lstrip("\t ")
            indent = len(l) - len(stripped)
            if "\t" in l[:indent]:
                depth = l[:indent].count("\t")
            else:
                depth = indent // max(1, unit)
            name = stripped.strip()
            # pop until we find a strictly shallower node = the real parent
            while stack and stack[-1][0] >= depth:
                stack.pop()
            parent = stack[-1][1] if stack else root
            node = Node(name, parent=parent)
            parent.children.append(node)
            stack.append((depth, node))
        return root

    # 3. flat list — one name per line
    for l in lines:
        root.add([l.strip()])
    return root


def walk(node: Node):
    """Yield (path, node) for every node, depth-first."""
    for c in node.children:
        yield c.path(), c
        yield from walk(c)


def leaves(node: Node) -> list[tuple[str, ...]]:
    """Paths of nodes with no children — the species-level entries."""
    out: list[tuple[str, ...]] = []
    for path, n in walk(node):
        if not n.children:
            out.append(path)
    return out


def is_category(name: str) -> bool:
    """True for structural nodes that are not species names."""
    n = name.strip().lower()
    if n in CATEGORY_HINTS:
        return True
    if _RANK_WORDS.match(n):
        return True
    return False


def species_names(node: Node, drop_categories: bool = True) -> list[str]:
    """Leaf names, optionally dropping category/rank nodes.

    Note: a leaf that IS a category (e.g. an empty 'Crabs' node) is dropped —
    scoring 'Crabs' as a taxon would produce confident nonsense.
    """
    out, seen = [], set()
    for path in leaves(node):
        name = path[-1].strip()
        if not name or name in seen:
            continue
        if drop_categories and is_category(name):
            continue
        seen.add(name)
        out.append(name)
    return out


# --- leaf convention: "common name: another common: Genus species" ---------
# The scientific name is the LAST segment that looks binomial; everything before
# it is a common name. A leaf like "Brown Algae: Brown Seaweeds: Phaeophyta" is a
# TAXON node, not a species, and must not become a candidate.
_BINOMIAL = re.compile(
    r"^[A-Z][a-zA-Z\u00c0-\u024f\-]+ "
    r"(?:[a-z][a-zA-Z\u00c0-\u024f\-]{1,}|sp\.|spp\.|cf\.|aff\.|forma)(?:\s|$)")

# Words that are never Latin epithets. Without this, English category phrases
# pass a naive Genus-species test: "Rays and Skates" reads as Genus "Rays" +
# epithet "and", and would be scored by BioCLIP as though it were a species.
# NOTE: taxonomic qualifiers (sp., spp., cf., aff., forma) are legitimate and
# must NOT be listed here — "Sargassum sp." is a real, usable genus-level entry.
_ENGLISH_STOP = {"and", "or", "of", "the", "in", "on", "with", "for", "a", "an",
                 "to", "from", "at", "by", "as", "misc", "other", "unknown",
                 "not", "no", "all", "any", "some", "its", "their"}


def looks_binomial(s: str) -> bool:
    """True when `s` plausibly is 'Genus species' / 'Genus sp.' and not English."""
    toks = s.split()
    if not 2 <= len(toks) <= 4:            # 4 allows "Genus species forma x"
        return False
    if not re.fullmatch(r"[A-Z][a-zA-Z\u00c0-\u024f\-]+", toks[0]):
        return False
    for t in toks[1:]:
        if t.lower() in _ENGLISH_STOP:
            return False
        if not re.fullmatch(r"[a-z][a-zA-Z\u00c0-\u024f\-]{1,}\.?", t):
            return False
    return True


# Sections that are tag dumps or unfinished, not species lists.
JUNK_SECTIONS = {"[more keywords]", "[needs more]", "[needs more keywords]",
                 "keywords", "other keywords"}


def split_leaf(name: str) -> tuple[list[str], str | None]:
    """Split a leaf into (common names, scientific name-or-None)."""
    segs = [s.strip() for s in name.split(":") if s.strip()]
    if not segs:
        return [], None
    if _BINOMIAL.match(segs[-1]) and looks_binomial(segs[-1]):
        return segs[:-1], segs[-1]
    return segs, None


def species_entries(node: Node, drop_junk: bool = True) -> list[dict]:
    """Extract species records: {scientific, common[], path}.

    Merges duplicates by scientific name and unions their common names, because
    these lists repeat a species once per common-name ordering (e.g. "bull kelp:
    bullwhip kelp" and "bullwhip kelp: bull kelp" are the same species).
    """
    by_sci: dict[str, dict] = {}

    def rec(n: Node, in_junk: bool) -> None:
        junk = in_junk or (drop_junk and n.name.strip().lower() in JUNK_SECTIONS)
        common, sci = split_leaf(n.name)
        if sci and not junk:
            e = by_sci.setdefault(sci, {"scientific": sci, "common": [],
                                        "path": n.path()[:-1]})
            for c in common:
                if c not in e["common"]:
                    e["common"].append(c)
        for c in n.children:
            rec(c, junk)

    rec(node, False)
    return list(by_sci.values())


def path_taxon_keys(entry: dict) -> list[str]:
    """The taxonomic names in a path — what narrowing should match against.

    Each node in these lists often carries both a plain-English label and a taxon
    name, e.g. "Sponge Crabs: Dromiidae" or "Nudibranchs: Nudibranchia". Only the
    taxon name is safe to match:
      - "Sponge Crabs" would otherwise be hit by a search for "sponge", and
        Dromiidae are CRABS — a "sponges" list would silently gain crabs;
      - "Coral Breams: Nemipteridae" would be hit by "coral", but they are FISH;
      - "False Morays: Chlopsidae" would be hit by "ray" (inside "mo-rays"), but
        they are EELS, not rays.
    So take the post-colon segment where there is one, else the node itself.
    """
    keys = []
    for p in entry.get("path", []):
        if ":" in p:
            tail = p.split(":")[-1].strip()
            if tail:
                keys.append(tail)
        else:
            keys.append(p.strip())
    return keys


def names_under_entries(entries: list[dict], pattern: str,
                        taxon_only: bool = True) -> list[str]:
    """Narrow extracted species to those whose PATH matches `pattern`.

    Defaults to matching taxon names only (see `path_taxon_keys`) and requires a
    whole-word match, because plain-English substring matching over a hierarchy
    produces confidently wrong groups.
    """
    rx = re.compile(rf"\b(?:{pattern})\b", re.I)
    out = []
    for e in entries:
        keys = path_taxon_keys(e) if taxon_only else list(e.get("path", []))
        if any(rx.search(k) for k in keys):
            out.append(e["scientific"])
    return out


def names_under(node: Node, pattern: str) -> list[str]:
    """Species under any node whose name matches `pattern` (regex, case-insens).

    This is the narrowing operation: names_under(tree, "crustacean") returns just
    the crustaceans instead of the whole multi-thousand-entry list.
    """
    rx = re.compile(pattern, re.I)
    keep: set[str] = set()
    for path, n in walk(node):
        if rx.search(n.name) or any(rx.search(p) for p in path[:-1]):
            keep.update(species_names(n))
    return [s for s in species_names(node) if s in keep]


def outline(node: Node, max_depth: int = 3, min_count: int = 1) -> str:
    """Printable hierarchy summary so a list can be inspected before use."""
    lines: list[str] = []

    def rec(n: Node, d: int):
        if d > max_depth:
            return
        kids = n.children
        if d > 0 and len(leaves(n)) < min_count:
            return
        if d > 0:
            lines.append("  " * (d - 1) + f"{n.name}  ({len(leaves(n))})")
        for c in sorted(kids, key=lambda x: -len(leaves(x)))[:12 if d >= 2 else 40]:
            rec(c, d + 1)

    for c in sorted(node.children, key=lambda x: -len(leaves(x))):
        rec(c, 1)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        tree = parse(open(sys.argv[1], encoding="utf-8", errors="replace").read())
        tot = leaves(tree)
        print(f"nodes: {len(list(walk(tree)))}   leaves: {len(tot)}   "
              f"species-ish: {len(species_names(tree))}")
        print("\n" + outline(tree, max_depth=int(sys.argv[2]) if len(sys.argv) > 2 else 3))
        if len(sys.argv) > 3:
            hits = names_under(tree, sys.argv[3])
            print(f"\nnarrowed to /{sys.argv[3]}/: {len(hits)} names")
            print(", ".join(hits[:25]))
