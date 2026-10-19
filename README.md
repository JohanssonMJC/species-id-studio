# Species ID Studio

A desktop app for identifying an organism from a photo, offline-first, with a
candidate list you control.

Two engines work together:

- **Tier 1 — BioCLIP 2 (local, in-process).** A zero-shot CLIP-family model that ranks
  a *supplied* list of candidate names against your photo. It cannot invent a name that
  is not in the list, which is the point: you constrain the answer space instead of
  trusting a fixed label set.
- **Tier 2 — a local vision LLM (optional).** A second opinion that describes
  diagnostic features and argues for a species. **Any vision model your `llama.cpp`
  build supports works** — `Qwen3-VL` is only the tested default, see
  [Choosing a vision model](#choosing-a-vision-model). Kept independent of Tier 1 by
  default (see [Two opinions, not one](#two-opinions-not-one)).

Around that sits the part that actually decides accuracy: **where the candidate list
comes from**. A photo can only be identified against names you give it.

![Species ID Studio identifying a Nembrotha chamberlaini nudibranch: ranked candidates on the right, the photo and the 880-name candidate list on the left](docs/screenshot.png)

<sub>Screenshot produced by `docs/make_screenshot.py` using the real model and the shipped
nudibranch list — the probabilities are BioCLIP's actual output, not mock-ups.</sub>


---

## Contents

- [Why it is built this way](#why-it-is-built-this-way)
- [Install](#install)
- [Models](#models)
- [Quick start](#quick-start)
- [Candidate sources](#candidate-sources)
- [Two opinions, not one](#two-opinions-not-one)
- [Photo metadata](#photo-metadata)
- [Batch mode](#batch-mode)
- [Building your own name lists](#building-your-own-name-lists)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Data sources and credit](#data-sources-and-credit)

---

## Why it is built this way

The core finding, from measurement rather than assumption:

> **Supplying a region- and group-specific candidate list is a bigger accuracy lever
> than changing the model.**

Two different kinds of list were compared on the same photos. A **global** one — all
7,405 marine names from a keyword export — and a **regional** one — the nudibranchs
recorded within 0–40 m of a single dive site, 393 at Anilao. Both put the right answer at
rank 1, so a long list does not degrade the ranking itself. What the long list costs is
text-encoding time, and what it dulls is the evidence behind the top pick: on one shrimp
the gap to the runner-up fell from 100% to 56%, though it still ranked first.

Scaling test over lists of 200 to 1,516 names — encoding grows roughly linearly,
per-photo scoring is flat, and top-1 never slipped:

| candidates | text encode | per-photo score | top-1 |
|---|---|---|---|
| 200 | 12 s | 0.49 s | 5/5 |
| 500 | 31 s | 0.49 s | 5/5 |
| 1,000 | 63 s | 0.49 s | 5/5 |
| 1,516 | 96 s | 0.49 s | 5/5 |

So the app is built around **narrowing**, not around a bigger model. Narrowing has a
failure mode of its own, though: narrowing by *group* must never drop the true species,
which is why each group maps to the widest taxon GBIF accepts (see
[Groups are deliberately broad](#groups-are-deliberately-broad)). Excluding the right
answer costs the entire identification; including extra names costs only some confidence.

---

## Install

Requires Python 3.11+ and a desktop session (Qt). Linux is what it was developed on;
macOS and Windows should work but are untested.

```bash
git clone https://github.com/<you>/species-id-studio.git
cd species-id-studio

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`uv` works too and is faster:

```bash
uv venv .venv && uv pip install --python .venv -r requirements.txt
```

Launch:

```bash
python species_gui.py
```

Check the engines without opening the window:

```bash
python species_gui.py --selftest
```

> `--selftest` only verifies that the engine files and config are present. It does
> **not** build the window, so it can pass while the UI is broken — the tests do that.

---

## Models

Model weights are **not** in this repository (they are gigabytes). Put them here:

| What | Path | Size | Needed for |
|---|---|---|---|
| BioCLIP 2 checkpoint | `~/Models/speciesid/bioclip-2/open_clip_model.safetensors` | ~2.3 GB | Tier 1 identification |
| BioCLIP 2 config | `~/Models/speciesid/bioclip-2/open_clip_config.json` | small | Tier 1 identification |
| Vision LLM | `~/Models/qwen_image/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf` | ~5 GB | Tier 2 second opinion |
| Vision projector | `~/Models/speciesid/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` | ~0.75 GB | Tier 2 second opinion |

Paths are overridable in the app (`Vision server → …`) and in
`~/.config/species_studio.json`.

The vision LLM is served by a `llama-server` the app owns (it starts and stops it
itself, on `127.0.0.1:8081`), which is why a `llama.cpp` build with `--mmproj` support
is required for Tier 2. **Tier 1 needs none of that** — BioCLIP runs in-process on CPU
at roughly 0.3 s per photo.

`open_clip` loads a local model from a directory containing
`open_clip_config.json` + `open_clip_model.safetensors`. If loading fails, that pair is
the first thing to check.

### Choosing a vision model

Nothing in the code is specific to one model. Drop a weights file and its projector into
`~/Models` and the app finds the pair, so the vision picker offers it:

```
~/Models/<anything>/
    <model>.gguf        # the language model
    mmproj-*.gguf       # its vision projector — REQUIRED, and must match the model
```

It works because the conversation is plain: one photo, one written question. There is no
model-specific prompt engineering to break. Two things do change between models:

- **Which architectures are loadable.** That is decided by your `llama.cpp` build, not by
  this app. The `tools/mtmd/models/` directory in your checkout lists exactly what your
  build can project. A model whose architecture is missing fails at load.
- **How much memory it wants.** Model and projector must both fit in GPU memory. A dense
  30B on an integrated GPU is slow for a single-photo question even when it fits, where a
  mixture-of-experts model of the same size is far faster — only a fraction of its
  parameters are read per token.

Measured sizes (Q4, weights plus projector): `Qwen3-VL-8B` ≈ 5.4 GB,
`Qwen3-VL-30B-A3B` ≈ 17.5 GB, `Kimi-VL-A3B-Thinking` ≈ 10.4 GB,
`Qwen3-VL-32B` (dense) ≈ 19.5 GB.

The projector must match the model. An `mmproj` from a different family can load and
produce nonsense, which is worse than refusing to start.

---

## Quick start

1. **Open a photo** (or drag one onto the window).
2. **Say where you were.** Either type a place in `Place` and press *Find place*, or
   type latitude/longitude directly (a GPS-tagged photo fills these itself).
3. **Say what kind of animal it is** in the `group` selector. You rarely know the
   species; you usually know the group, and the group decides the candidate list.
4. **Load a candidate list** — `Place list (GBIF)`, `Marine list…`, or
   `Load from file…` (see below).
5. **Identify (local BioCLIP)** → the ranked list appears, with probabilities.
6. Optionally **Ask vision model** for a second opinion.
7. **Copy a name** with `Ctrl+C` or right-click → *Copy name*
   / *Open in iNaturalist* / *Open in GBIF* / *Open in Google Images*.

The status line always shows **where the next lookup will search** — coordinates,
place name and radius — because a wrong location silently produces a plausible but
wrong species list.

---

## Candidate sources

The list box is the answer space. Three ways to fill it:

### `Place list (GBIF)` — what is actually recorded near you

Queries GBIF occurrence records around the coordinates, narrowed by group and
habitat, and ranks species by **how often they have been recorded there**.

```
GBIF: 14,173 occurrence records | ~100 km around 13.715,120.896
      | taxon Nudibranchia | 541 species
```

(Occurrence counts grow as GBIF receives records, so the record total will not match
exactly; the species count for a fixed area and filter is much steadier.)

#### There is no depth filter at all, on purpose

GBIF has no habitat switch: `isMarine`, `habitat` and `isTerrestrial` do not exist on
`/v1/occurrence/search` and are silently ignored. What exists is `depth`, which selects
the records that happen to **carry a depth reading** — and depth is reported
inconsistently. A depth filter can therefore only ever subtract, in two different ways:

| Anilao, 100 km | no depth filter | "has a depth reading" | + a 0–40 m band |
|---|---|---|---|
| nudibranchs | 541 | 398 | 393 |
| crustaceans | 1,672 | 1,100 | 220 |
| echinoderms | 668 | 533 | 192 |
| corals & anemones | 458 | 197 | 63 |

- **Records with no depth value are dropped by any depth filter.** 143 of the 541
  nudibranch species at Anilao are recorded there with no depth at all. They are not deep
  animals — their records lack the field.
- **Species recorded only deeper than the band are dropped too.** Small for nudibranchs
  (5 species), but at the same site **880 of 1,672 crustacean species and 341 of 668
  echinoderm species are recorded only below 40 m**. A 0–40 m band would show 220
  crustaceans out of 1,672, which reads as "the list is broken" when in fact the filter
  is doing exactly what it was asked.

So **the app has no depth control** — not even an "underwater only" switch. Narrowing by
*taxon*, broadly, is safe; narrowing by *depth* silently removes real species, and a
control that only ever subtracts is a trap rather than a feature. The result is the most
complete list GBIF can give you for that area, and the group selector keeps the taxa
right.

`fetch_candidates(depth=...)` still exists for scripts that want a range, and the CLI
takes `--depth MIN,MAX` with that warning in its help text. The GUI never passes it.

- **No account or API key required.**
- **Loading replaces the list** (there is an `add to list` checkbox if you want to
  merge instead).
- The count label names the taxon, e.g. `393 candidates — Nudibranchia`, so you can
  see the filter is applied.

#### Groups are deliberately broad

A candidate list that silently excludes the right answer gives the model a **zero**
chance of finding it, and no ranking can recover that. So every group maps to the
widest taxon that GBIF will accept, and where no single key covers a group the
queries are unioned:

```
crustaceans   Malacostraca + Maxillopoda + Copepoda + Branchiopoda + Ostracoda + Remipedia
worms         Annelida + Nemertea + Sipuncula + Chaetognatha
jellyfish     Scyphozoa + Cubozoa + Hydrozoa + Ctenophora
tunicates     Ascidiacea + Thaliacea + Appendicularia
algae_seaweed Rhodophyta + Ochrophyta + Chlorophyta
```

Earlier versions quietly mapped *crustaceans* to `Decapoda` (crabs only) and
*echinoderms* to `Asteroidea` (starfish only). At Anilao, 0–40 m that returned **39**
echinoderm species against **192** for the phylum, and **159** crustaceans against
**220** — mantis shrimps, krill, barnacles and isopods were all unreachable.

Two GBIF traps make this less obvious than it sounds: `Crustacea` and `Tunicata` each
match a real **genus** of that name, and several natural broad names do not exist in
the backbone at all (`Heterobranchia`, `Chondrichthyes`, `Actinopterygii`, `Teleostei`,
`Sacoglossa`). Genus- and species-rank matches are therefore rejected outright, not
just non-`EXACT` ones.

If in doubt, choose **`no group — I don't know`**: that applies no taxon filter and
returns everything recorded near you. A longer list costs a little ranking accuracy;
a short wrong list costs the identification.

### `Marine list…` — your own curated list, offline

**The name lists are not shipped here** — they are derived from *MarineLife Keywords
1.0* (2009), a commercial product this project does not redistribute. If you own a copy,
one command rebuilds them locally:

```bash
python make_marine_lists.py "/path/to/MarineLife Keywords 1.0.txt"
```

Reference run: 12,601 keyword records → 8,696 unique species → **7,405 distinct usable
names**, grouped so you can load the ~880 nudibranchs instead of all 7,405 marine
species. Synonymy collapses some records onto one accepted name; 632 genus-level and
226 unmatchable names are kept apart rather than guessed. Works with no network.
Full detail and options: [`docs/REGENERATE.md`](docs/REGENERATE.md).

### `Load from file…` — one scientific name per line

Any plain text list. Comments (`#`) and blank lines are ignored.

---

## Two opinions, not one

`Ask vision model` sends the photo to the local vision LLM. By default it is **blind**
to BioCLIP's answer, and that is deliberate.

The answer lands in the *Vision model reasoning* tab as plain text you own: select any
part and press `Ctrl+C`, or right-click for **Copy selection**, **Copy whole answer**,
and — when the selection looks like a binomial — direct links to look that name up in
iNaturalist, GBIF or Google Images. The model writes prose rather than rows, so a name
it mentions in passing is exactly what you want to check.

The answer lands in the *Vision model reasoning* tab, and it is plain text you own:
select any part and press `Ctrl+C`, or right-click for **Copy selection**, **Copy whole
answer**, and — when the selection looks like a binomial — direct links to look that
name up in iNaturalist, GBIF or Google Images. The model writes prose rather than rows,
so a name it mentions in passing is exactly what you want to check.

Observed in real use: BioCLIP put the correct species at rank 10 of ~540, and a vision
model that had been shown a candidate list simply restated rank 1 — which was wrong.
A second opinion is only worth asking for if it is independent.

So:

- **The candidate list is opt-in.** A checkbox — *"Give the vision model the BioCLIP
  list (biases it)"* — is off. Unticked, the model sees only the photo.
- **It reasons before naming.** The prompt demands diagnostic features first, then a
  name, and explicitly allows "not enough detail to name a species".
- **It gets context, but not answers.** Location, habitat and the assumed group are
  always included, framed as fallible:

  ```
  Context supplied by the user. It is a hint, not evidence, and it may be wrong —
  if the photo clearly contradicts it, say so and go with what you can see:
    - location: 13.715, 120.896 (Mabini, PH, BT), within roughly 100 km
    - habitat: underwater (marine records only)
    - assumed group (the user's own guess): sea · nudibranchs & sea slugs — i.e. Nudibranchia
  ```

- **Afterwards the app compares the two** and says where the vision answer sat in
  BioCLIP's ranking:

  ```
  cross-check: the vision model named X, which BioCLIP ranked #10 (1.4%)
               — the two disagree on the top pick
  cross-check: the vision model agreed with BioCLIP's top pick (Y, 69.5%)
               — not independent evidence, and if that pick is wrong both are wrong
  ```

`top_k` (how many ranks are shown) defaults to **15**; raise it in
`~/.config/species_studio.json`.

---

## Photo metadata

EXIF is read when a photo is opened or batched:

| Field | Used for |
|---|---|
| GPS | fills Lat/Lon, which drives the regional candidate list |
| Date | reported (month filtering is available but off by default — see below) |
| Make/Model | reported |

**Month filtering is off by default.** Enabling it is expensive: the same Anilao query
returns 1,357 records / 314 species unfiltered and 2 records in January. GBIF coverage
is sparse and survey-driven per month, so a month filter measures when people happened
to visit, not seasonality.

Most web images carry no metadata (in one real 22-image set: 9 had EXIF, 8 a date, 0
GPS). The app says so rather than failing silently.

---

## Batch mode

- **`Batch folder…`** — every image in a folder (optional recursion).
- **`Batch files…`** — pick photos by hand (ctrl-click for several).

Each photo is identified with **its own GPS and date**, so a set shot at one dive site
gets that site's list per photo. Results are auto-saved as CSV **and** JSON to
`~/species_results/`, and can be exported again from the Batch tab.

**Selecting a row previews the photo** in the viewer on the left — filenames like
`DSC_4821.NEF` tell you nothing about the animal. The preview deliberately does *not*
apply EXIF, so clicking through a batch cannot silently rewrite your search area.

Columns: photo, taken, GPS, most likely species, probability, candidate source.

---

## Building your own name lists

`make_marine_lists.py` runs all four stages in order — this is the command to use:

```bash
python make_marine_lists.py "/path/to/MarineLife Keywords 1.0.txt"
```

Under the hood it is four scripts, kept separate because the same four problems bite
every time and you may want to re-run one of them alone:

| stage | script | why it is its own step |
|---|---|---|
| parse | `lightroom_keywords.py` | the export is CR-only with tab-depth nesting; get this wrong and every name is garbage |
| modernise | `modernize_marinelife.py` | GBIF renames; needs a 45-minute resumable cache |
| repair | `repair_typos.py` | typos are dangerous — this refuses to guess a different species |
| emit | `build_mkw_candidates.py` | grouping is a judgement call, not a lookup |

```bash
python lightroom_keywords.py "MarineLife Keywords 1.0.txt"   # outline only
python modernize_marinelife.py --src marine_lists/marinelife_entries.json \
                               --cache marine_lists/mkw_gbif_cache.json \
                               --out marine_lists/marinelife_modern.json
python repair_typos.py --modern marine_lists/marinelife_modern.json \
                       --cache marine_lists/mkw_typo_cache.json
python build_mkw_candidates.py --dir marine_lists
```

Result: `marine_lists/*.txt`, one file per group, plus
`_UNRESOLVED_review_these.txt` (226 names kept rather than guessed) — and
`_genus_level.txt` for names only identified to genus.

**The typo rule is the important part.** A name is only "repaired" if it is within 2
edits per token and 3 total, and the result is confirmed to resolve to an active taxon.
`Chromi alta → Chromis alta` is accepted (1 edit). `Trapezia wardi → Trapezia tigrina`
is refused (a different crab, 6 edits). Every repair records from → to, distance, path
and evidence.

---

## Tests

```bash
python tests/test_gbif_naming.py    # naming rules, group mapping, geometry
python tests/test_typo_rules.py     # the typo safety predicate
python tests/test_lightroom.py      # keyword-export parsing
```

These three are pure logic — no network, no models, no Qt — and are what CI runs.

The full suite used during development (17 files) additionally covers the GUI end to
end, but needs PyQt6, the model files and network access, so it is not in CI. If you add
GUI tests, run them offscreen:

```bash
QT_QPA_PLATFORM=offscreen SPECIES_CFG=/tmp/test-cfg.json python tests/test_gui.py
```

**Always set `SPECIES_CFG` in tests.** It redirects the config file so a test cannot
rewrite your real settings — this is enforced by checking the config's hash before and
after a suite run.

---

## Known limitations

Stated plainly, because each one is a silent-wrong-answer if you do not know about it.

- **"Reef fish" cannot be filtered in GBIF.** `/species/204` *is* Actinopterygii, but
  every fish occurrence record carries `classKey = None`, so `taxonKey=204` returns 0
  records. The group therefore falls back to restricting the regional list against your
  own bony-fish names, and says so. For fish, the iNaturalist path
  (`Fill from location`) is better — it has a working Actinopterygii taxon.
- **There is no "land only" filter.** GBIF has no `isTerrestrial` (nor `isMarine`).
  Leaving depth unset does not select land — it stops excluding marine. Depth presence
  is *metadata*, not habitat: many marine records simply have no depth value.
- **Counts measure survey effort, not abundance.** 422 *Chromodoris annae* at Anilao
  means 422 catalogue records, not that it is the commonest animal on the reef.
- **Place names are ambiguous.** "Anilao" matches exactly one iNaturalist place —
  Anilao, *Iloilo*, ~1,000 km from the Batangas dive area, where only 4 nudibranch
  species are recorded. The dive area is filed as **Mabini, PH, BT (13.715, 120.896)**.
  The app asks you to confirm coordinates when a name matches only one place, for
  exactly this reason.
- **BioCLIP ranks plausibility, not certainty.** A hard species in a species-rich genus
  lands mid-list. It is a zero-shot image-text matcher, not a taxonomist.
- **The window's minimum width is ~1,300 px.** Below that it will overhang a small
  display; the layout cannot shrink further without restructuring the control rows.
- **`--selftest` does not build the window.** It checks engine files only.

---

## Data sources and credit

- **GBIF** — occurrence and backbone taxonomy via `https://api.gbif.org/v1`.
  Occurrence data is CC BY 4.0 / CC0 depending on the dataset; see
  <https://www.gbif.org/citation-guidelines>. This app queries the public API live and
  does not redistribute GBIF records.
- **iNaturalist** — place search and regional species lists via
  `https://api.inaturalist.org/v1`. Observations are individually licensed; the API is
  queried live.
- **BioCLIP 2** — image encoder from the BioCLIP project (Stevens et al., CVPR 2024;
  BioCLIP 2: Gu et al.). Check the model card for its licence before redistribution.
- **Qwen3-VL** — vision-language model by Alibaba Qwen, Apache-2.0. This is the
  **tested default for Tier 2, not a dependency**: any vision model your `llama.cpp`
  build supports can be used instead, and its licence is then yours to check.
- **MarineLife Keywords 1.0** (2009) — *MarineLife Keywords* was used to guide the
  marine candidate lists this project was developed against. **Neither the product nor
  any file derived from it is included in this repository.** `make_marine_lists.py` can
  consume a copy **you own** to build lists on your own machine; how you may use that
  copy is between you and its publisher, and nothing here grants or implies a licence to
  it. This project is not affiliated with, sponsored by, or endorsed by the publisher.

  What this repository does contain is its own work: the parsing, the GBIF
  modernisation, the typo-repair rules and the grouping (§ *Building your own name
  lists*). Taxonomic names are facts, and every name in a rebuilt list has been resolved
  against the GBIF backbone above, under GBIF's terms.

If you publish identifications made with this tool, cite GBIF and the underlying
observation datasets — a species list is only as good as the records behind it.

---

## License

MIT — see [LICENSE](LICENSE). **No name lists are bundled**, so nothing derived from a
third-party product is redistributed here. Third-party models keep their own terms; see
[Data sources and credit](#data-sources-and-credit).
