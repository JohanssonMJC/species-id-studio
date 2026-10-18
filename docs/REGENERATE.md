# Regenerating the marine candidate lists

The `marine_lists/*.txt` name lists are **not** in this repository. They are derived
from *MarineLife Keywords 1.0* (2009), a commercial keyword product, and this project
does not redistribute it or its derived files.

If you own a copy, one script rebuilds exactly the same lists locally.

```bash
python make_marine_lists.py "/path/to/MarineLife Keywords 1.0.txt"
```

Accepts the `.txt` or the `.zip` it arrived in. Output goes to `./marine_lists/`,
which the app already looks in.

---

## What it does

| stage | what happens | needs | time |
|---|---|---|---|
| 1. parse | the keyword export → species records (`marinelife_entries.json`) | nothing | seconds |
| 2. modernise | every name resolved against the GBIF backbone | network | ~45 min uncached |
| 3. repair | residual single-character typos, strictly and verified | network | a few minutes |
| 4. emit | one `.txt` per group for the GUI | nothing | seconds |

For the reference run: 12,601 keyword records → **8,696 unique species** → **7,405
distinct usable names**. (Fewer than 8,696 because synonymy collapses several old names
onto one accepted name.) Two review files are deliberately *not* guessed:

- `_UNRESOLVED_review_these.txt` — 226 names GBIF cannot match even after typo repair
- `_genus_level.txt` — 632 names only identified to genus

Measured group sizes from that run:

| group | names | group | names |
|---|---|---|---|
| bony_fishes | 3,174 | worms | 121 |
| sea_slugs | 938 | sponges | 94 |
| nudibranchs | 880 | tunicates | 81 |
| cephalopods | 608 | cetaceans | 81 |
| crustaceans | 435 | algae_seaweed | 72 |
| sharks_rays | 405 | jellyfish | 57 |
| corals_anemones | 316 | sea_snakes | 48 |
| crabs_shrimps | 283 | bryozoans | 34 |
| echinoderms | 261 | seals | 31 |
| | | turtles / dugongs_manatees | 5 each |

`all_marine.txt` (7,405) is the union; the 21 group files overlap by design.

## Options

```bash
# check the export parsed correctly, without any network calls
python make_marine_lists.py "MarineLife Keywords 1.0.txt" --skip-modernise

# smoke-test the first 100 names (~30 s)
python make_marine_lists.py "MarineLife Keywords 1.0.txt" --limit 100 --out /tmp/try

# reuse a cache from a previous run (or from another machine) — lookups are free
python make_marine_lists.py "MarineLife Keywords 1.0.txt" --cache ./mkw_gbif_cache.json

# be gentler on the GBIF API
python make_marine_lists.py "MarineLife Keywords 1.0.txt" --delay 0.5
```

## It is safe to interrupt

Stages 2 and 3 cache **every** lookup to `marine_lists/mkw_gbif_cache.json` and
`marine_lists/mkw_typo_cache.json`. Kill the run at any point and start it again — it
picks up where it stopped. A completed rebuild run again costs nothing but a few
seconds of file I/O.

Rate: unauthenticated GBIF allows roughly 3 lookups/second, which is where the ~45
minutes for 8,696 names comes from. No account or API key is needed.

## What the names become

Stage 2 keeps an accepted, current binomial and records how it got there:

| status | meaning |
|---|---|
| `current` | already the accepted name |
| `renamed` | a synonym; replaced with the accepted name (*Chromodoris binza → Felimida binza*) |
| `typo-fixed` | GBIF's own fuzzy match, confirmed |
| `typo-repaired` | recovered locally and verified (see below) |
| `genus-level` | only identified to genus — kept apart, never presented as a species |
| `unresolved` | no match anywhere — kept in a review file rather than guessed |

Stage 3 repairs typos under a deliberately narrow rule, because the failure mode that
matters is substituting **a different species**:

- each differing token within **2** edits, and **3** edits total; or
- identical epithet with the genus replaced, then confirmed against GBIF;
- always requires an **active** taxon, and always records from → to, distance and
  evidence.

```
ACCEPTED   Chromi alta          → Chromis alta            (1 edit)
ACCEPTED   Anoplopima fimbria   → Anoplopoma fimbria      (1 edit)
REFUSED    Trapezia wardi       → Trapezia tigrinа         (a different crab, 6 edits)
```

It also rejects genus-rank records masquerading as species — `Dermatobranchus van
Hasselt, 1824` is a genus-level entry whose author begins with the Dutch particle
"van", and stripping it naively produced the fake species *Dermatobranchus van*.

## Verifying the rebuild

Parsing, grouping and emission are deterministic; **stage 2 is not**, because GBIF's
backbone is a living checklist. A rebuild from a cache captured years ago, or against
today's backbone, can differ from an older run by a name or two.

Measured on the reference rebuild: **19 of 23 files byte-identical**; the four that
differed were single names moving between buckets —

```
all_marine      7,405 → 7,405   Comaster schlegelii ↔ Staurostoma mertensii
_UNRESOLVED       228 → 226
echinoderms       262 → 261
jellyfish          56 →  57
```

That is the expected shape of the drift: one name resolved differently, so it left the
unresolved file and entered `echinoderms`/`jellyfish` instead. If you are comparing
against a known-good copy, expect this rather than a perfect match, and check `diff` for
*which* names moved before assuming a bug:

```bash
diff <(sort known_good/nudibranchs.txt) <(sort marine_lists/nudibranchs.txt)
```

## If you do not have the original

You do not need it. The app's other candidate sources work on their own:

- **`Place list (GBIF)`** — species actually recorded near a place, by group and
  habitat. Nothing to install.
- **`Fill from location`** — an iNaturalist summary for a place and month.
- **`Load from file…`** — any plain-text list of names, one per line.
