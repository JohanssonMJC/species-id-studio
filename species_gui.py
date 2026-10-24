#!/usr/bin/env python3
"""
Species ID Studio — local-first photo identification for wildlife.

Two engines behind one window, both able to run with no network:

  Tier 1  BioCLIP 2        in-process, CPU, 0.3 s/image. Zero-shot taxonomic
                           ranking against a candidate list you supply. This is
                           the classifier — it is what makes "possible species"
                           meaningful rather than one forced answer.
  Tier 2  Qwen3-VL-8B      served by a llama-server instance this app owns, on
                           its own port with --mmproj. Describes diagnostic
                           features and sanity-checks Tier 1.

Online is OPTIONAL and off by default: when you enable it, iNaturalist is used
to fill the candidate list with species actually recorded near a location in a
given month. That geo/season filter is the single biggest accuracy lever — most
confusions are geographically impossible lookalikes.

The vision server is deliberately NOT your systemd llama-server: that one is
configured for your chat model. This app starts its own instance on :8081 and
leaves your chat setup untouched.

Usage:  species_gui.py [--vision-port 8081] [--selftest]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import urllib.parse
import urllib.request

from PyQt6.QtCore import QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (QAction, QColor, QDesktopServices, QGuiApplication, QIcon,
                         QKeySequence, QPixmap, QShortcut)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QSplitter,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser,
    QVBoxLayout, QWidget,
)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

try:
    import speciesid
except Exception as exc:                       # pragma: no cover
    speciesid = None
    print(f"[warn] speciesid.py not importable: {exc}", file=sys.stderr)

try:
    import vision_models
except Exception as exc:                       # pragma: no cover
    vision_models = None
    print(f"[warn] vision_models.py not importable: {exc}", file=sys.stderr)

try:
    import gbif_candidates
except Exception as exc:                       # pragma: no cover
    gbif_candidates = None
    print(f"[warn] gbif_candidates.py not importable: {exc}", file=sys.stderr)

try:
    import personal_classifier
except Exception as exc:                       # pragma: no cover
    personal_classifier = None
    print(f"[warn] personal_classifier.py not importable: {exc}", file=sys.stderr)

APP_NAME = "Species ID Studio"
# Tests and throwaway runs must point this elsewhere via SPECIES_CFG. Writing
# test-derived locations into the real config makes the app start up pre-filled
# with a place the user never chose — a test would silently become their default.
CFG_PATH = Path(os.environ.get("SPECIES_CFG")
                or (Path.home() / ".config" / "species_studio.json"))
VISION_PORT = 8081
VISION_PIDFILE = Path.home() / ".cache" / "species_vision.pid"
VISION_LOG = Path.home() / ".cache" / "species_vision.log"
LLAMA_SERVER = Path.home() / "src" / "llama.cpp" / "build" / "bin" / "llama-server"
VISION_MODEL = Path.home() / "Models" / "qwen_image" / "text_encoders" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
VISION_MMPROJ = Path.home() / "Models" / "speciesid" / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
INAT_API = "https://api.inaturalist.org/v1"

DEFAULTS = {
    "vision_port": VISION_PORT,
    "online": False,
    "lat": "",
    "lon": "",
    "radius_km": 50,
    # 15, not 10: the correct species in a real run landed at rank 10 — the last
    # VISIBLE row — while ~540 candidates were scored. The cut-off was hiding
    # near misses, and the extra rows cost nothing (scoring is already done).
    "top_k": 15,
    "last_dir": str(Path.home() / "Pictures"),
    # Personal classifier: nearest-neighbour over BioCLIP embeddings of the user's own
    # labelled photos. Off until a folder has been fitted, because it can only answer for
    # the species it was given. 1-NN is the default: at one example per species it beat
    # the centroid average on held-out photos (75.0% vs 75.0% overall, 90.0% on species
    # with labels, against 65.0% for text ranking alone).
    "personal_enabled": True,
    "personal_use_centroids": False,
}


def personal_path() -> Path:
    """Fitted prototypes live beside the config, never inside the app or the repository:
    they are made of the user's photographs and are not distributable."""
    return CFG_PATH.parent / "personal_classifier.npz"


def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CFG_PATH.read_text()))
    except Exception:
        pass
    return cfg


def save_cfg(cfg: dict) -> None:
    try:
        CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CFG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def http_json(url: str, timeout: int = 30, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "species-studio/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


_region_cache: dict[tuple, list[str]] = {}

# How you actually think about a photo: you usually know the *place* and the
# rough *group* ("a nudibranch", "a warbler"), never the species. So the list is
# organised by habitat then group, and each entry resolves to real iNaturalist
# taxon ids rather than a coarse iconic rank.
#
# Why this matters, measured at Anilao (13.75, 120.93):
#   unfiltered  -> land AND reef species competing for the same 200 slots
#   Nudibranchia (47113) -> 120 species: Nembrotha chamberlaini, Taringa halgerda
#   Actinopterygii (47178) -> 140 species: Solenostomus paradoxus, pipefish
# Same place, completely different answer space. Picking the group is the lever.
GROUPS: list[tuple[str, dict]] = [
    ("no group — I don't know", {}),
    ("", {}),                                    # section separator

    ("land · birds", {"iconic": "Aves"}),
    ("land · insects", {"iconic": "Insecta"}),
    ("land · spiders & ticks", {"iconic": "Arachnida"}),
    ("land · mammals", {"iconic": "Mammalia"}),
    ("land · reptiles", {"iconic": "Reptilia"}),
    ("land · amphibians", {"iconic": "Amphibia"}),
    ("land · plants", {"iconic": "Plantae"}),
    ("land · fungi & lichens", {"iconic": "Fungi"}),
    ("", {}),

    # Marine megafauna. These MUST be their own "sea" entries rather than living
    # under land · reptiles / land · mammals: a diver photographing a green turtle
    # or a dolphin looks under sea, not under "reptiles". Note that iNaturalist has
    # no "Pinnipedia" taxon at all (searching it returns Phocoidea, a porpoise
    # superfamily), so seals are the union of their three real families.
    ("sea · turtles", {"taxa": [372234]}),               # Chelonioidea
    ("sea · dolphins & whales", {"taxa": [152871]}),     # Cetacea
    ("sea · seals & sea lions", {"taxa": [41736, 41687, 41764]}),  # Otariidae/Phocidae/Odobenidae
    ("sea · sea snakes", {"taxa": [492346]}),            # Hydrophiinae
    ("sea · dugongs & manatees", {"taxa": [46306]}),     # Sirenia
    ("sea · anything big in the water",                 # all of the above
     {"taxa": [372234, 152871, 41736, 41687, 41764, 492346, 46306]}),
    ("", {}),

    ("sea · reef fish", {"taxa": [47178]}),
    ("sea · sharks & rays", {"taxa": [47273]}),
    ("sea · nudibranchs & sea slugs", {"taxa": [47113]}),
    ("sea · crabs, shrimps & lobsters", {"taxa": [47187]}),
    ("sea · octopus, squid & cuttlefish", {"taxa": [47459]}),
    ("sea · corals, anemones & jellies", {"taxa": [47533]}),
    ("sea · starfish & sea urchins", {"taxa": [47668, 47548]}),
    ("sea · sea cucumbers", {"taxa": [47720]}),
    ("sea · clams, mussels & oysters", {"taxa": [47584]}),
    ("sea · sponges & sea squirts", {"taxa": [48824, 47811]}),
    ("sea · not sure which group", {"taxa": [47178, 47113, 47187, 47533]}),
    ("", {}),

    ("snails & slugs (land and sea)", {"taxa": [47114]}),
]

GROUP_LABELS = [label for label, _ in GROUPS]

# --- the user's own modernised marine keyword lists -------------------------
# NOT shipped (see docs/REGENERATE.md): built locally from a MarineLife Keywords 1.0
# (2009) export you own, by resolving every name against GBIF and keeping the accepted
# binomial. Grouped so a diver can pull in the ~880 nudibranchs rather than all ~7,400
# marine names: a narrower list encodes faster and keeps the top pick's confidence
# sharper (measured: a shrimp's margin fell 100% -> 56% between a 200- and a 1,516-name
# list, though it stayed #1).
MARINE_LISTS = Path(__file__).parent / "marine_lists"
MARINE_GROUPS: list[tuple[str, str]] = [
    ("all marine species", "all_marine.txt"),
    ("nudibranchs & sea slugs", "sea_slugs.txt"),
    ("nudibranchs only", "nudibranchs.txt"),
    # Every gastropod that is not a nudibranch: cones, cowries, ovulids, whelks, limpets,
    # sea hares. Without this group 1,353 names — a fifth of the whole list — could only
    # be reached through "all marine species".
    ("sea snails, cones & cowries", "gastropods.txt"),
    ("bivalves & chitons", "bivalves.txt"),
    ("other molluscs (aplacophorans)", "other_molluscs.txt"),
    ("octopus, squid & cuttlefish", "cephalopods.txt"),
    ("crabs, shrimps & lobsters", "crabs_shrimps.txt"),
    ("all crustaceans", "crustaceans.txt"),
    ("sea spiders", "sea_spiders.txt"),
    ("reef & bony fishes", "bony_fishes.txt"),
    ("sharks & rays", "sharks_rays.txt"),
    ("corals & anemones", "corals_anemones.txt"),
    ("starfish, urchins, brittle stars & sea cucumbers", "echinoderms.txt"),
    ("sponges", "sponges.txt"),
    ("tunicates (sea squirts, salps)", "tunicates.txt"),
    ("jellyfish, hydroids & comb jellies", "jellyfish.txt"),
    ("bristle worms & flatworms", "worms.txt"),
    ("bryozoans", "bryozoans.txt"),
    ("algae & seaweeds", "algae_seaweed.txt"),
    ("sea turtles", "turtles.txt"),
    ("crocodiles & marine iguana", "crocodiles.txt"),
    ("sea snakes", "sea_snakes.txt"),
    ("dolphins & whales", "cetaceans.txt"),
    ("seals, sea lions, otters & bears", "seals.txt"),
    ("dugongs & manatees", "dugongs_manatees.txt"),
]
# A short "per taxon" cap when a group unions several taxa, so the merged list
# stays balanced instead of being dominated by whichever group is commonest.
_SHORT_PER_TAXA = 60


def _one_region_query(params: dict) -> list[str]:
    d = http_json(f"{INAT_API}/observations?{urllib.parse.urlencode(params)}", timeout=45)
    names, seen = [], set()
    for obs in d.get("results", []):
        n = (obs.get("taxon") or {}).get("name")
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    return names


def resolve_places(name: str, limit: int = 8) -> list[dict]:
    """All places matching a name, so the user can disambiguate.

    Never silently pick one. Place names are genuinely ambiguous — "Mabini"
    alone matches four Philippine municipalities — and iNaturalist's place
    database does not always contain the name people use: the dive area
    "Anilao" exists as *Mabini, PH, BT*, while "Anilao" itself resolves to a
    different, tiny hamlet in Iloilo 1,000 km away. Picking the wrong one
    returns a plausible-looking but wrong species list, so the user must choose
    and must be able to see the coordinates they are choosing.
    """
    d = http_json(f"{INAT_API}/places/autocomplete?" +
                  urllib.parse.urlencode({"q": name, "per_page": limit}), timeout=30)
    out: list[dict] = []
    for r in d.get("results", [])[:limit]:
        lat = lon = None
        try:
            bb = r["bounding_box_geojson"]["coordinates"][0]
            lons = [c[0] for c in bb]
            lats = [c[1] for c in bb]
            lat, lon = round(sum(lats) / len(lats), 5), round(sum(lons) / len(lons), 5)
        except Exception:
            pass
        out.append({"id": r.get("id"), "name": r.get("display_name"),
                    "lat": lat, "lon": lon,
                    "obs": r.get("observations_count")})
    want = name.strip().lower()
    # exact first-word matches first (typing "Mabini" should not lead with
    # "Mabinay"), then everything else in iNaturalist's own order
    out.sort(key=lambda p: 0 if str(p["name"] or "").split(",")[0].strip().lower()
             == want else 1)
    return out


def fetch_regional_candidates(lat: str | float | None = None,
                              lon: str | float | None = None,
                              radius_km: int = 50, month: int | None = None,
                              spec: dict | None = None,
                              place_id: int | None = None) -> list[str]:
    """Species recorded near a point — or inside a named place — from iNaturalist.

    Pass either lat/lon (+radius) or place_id, plus an optional ``spec`` from
    GROUPS: ``{"iconic": "Aves"}`` or ``{"taxa": [47113]}``. Results are cached
    per location + radius + month + group, because in batch mode many photos
    share a location and re-querying per photo would be slow and rate-limit prone.
    """
    spec = spec or {}
    key = (place_id or 0,
           round(float(lat), 2) if lat not in (None, "") else 0,
           round(float(lon), 2) if lon not in (None, "") else 0,
           int(radius_km), month or 0,
           json.dumps(spec, sort_keys=True))
    if key in _region_cache:
        return _region_cache[key]

    base: dict = {"per_page": 200, "order_by": "votes",
                  "quality_grade": "research", "rank": "species"}
    if place_id:
        base["place_id"] = place_id
    else:
        base.update(lat=lat, lng=lon, radius=radius_km)
    if month:
        base["month"] = month

    # build one query per taxon so several taxa can be unioned
    queries: list[dict] = []
    if spec.get("taxa"):
        ids = list(spec["taxa"])
        short = len(ids) > 1
        for tid in ids:
            q = dict(base)
            q["taxon_id"] = tid
            if short:
                q["per_page"] = _SHORT_PER_TAXA
            queries.append(q)
    else:
        q = dict(base)
        if spec.get("iconic"):
            q["iconic_taxa"] = spec["iconic"]
        queries.append(q)

    merged, seen = [], set()
    for q in queries:
        try:
            for n in _one_region_query(q):
                if n not in seen:
                    seen.add(n)
                    merged.append(n)
        except Exception:
            if len(queries) == 1:
                raise
            continue                      # one taxon failing shouldn't kill the rest
    _region_cache[key] = merged
    return merged


def get_engine():
    """One BioCLIP instance per process — loading costs ~4 s."""
    if BioClipWorker._engine is None:
        if speciesid is None:
            raise RuntimeError("speciesid.py is missing from the app directory.")
        BioClipWorker._engine = speciesid.BioCLIP()
    return BioClipWorker._engine


# --------------------------------------------------------------------------- #
# Tier 1 — BioCLIP in a thread (model load is ~4 s, inference ~0.3 s)
# --------------------------------------------------------------------------- #
class GbifLookupWorker(QThread):
    """Ask api.gbif.org about one name, because the website cannot be opened for it."""
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def run(self) -> None:
        try:
            if gbif_candidates is None:
                self.failed.emit("gbif_candidates.py is missing from the app directory.")
                return
            self.done.emit(gbif_candidates.lookup_species(self.name))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PersonalFitWorker(QThread):
    """Embed every labelled photo in a folder and save the prototypes.

    Runs off the UI thread because a few hundred photographs is a minute of CPU work, and
    reuses BioClipWorker's engine so the model is not loaded a second time.
    """
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, folder: str, out_path: str):
        super().__init__()
        self.folder, self.out_path = folder, out_path

    def run(self) -> None:
        try:
            if speciesid is None or personal_classifier is None:
                self.failed.emit("speciesid.py or personal_classifier.py is missing.")
                return
            if BioClipWorker._engine is None:
                BioClipWorker._engine = speciesid.BioCLIP()
            meta = personal_classifier.fit(Path(self.folder), BioClipWorker._engine,
                                            Path(self.out_path))
            self.done.emit(meta)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BioClipWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    _engine = None                       # class-level cache: load once per process

    def __init__(self, image: str, candidates: list[str], top_k: int,
                 personal=None, use_centroids: bool = False):
        super().__init__()
        self.image, self.candidates, self.top_k = image, candidates, top_k
        # (prototypes, labels) or None. Passed in rather than loaded here so the worker
        # thread never touches the config file.
        self.personal = personal
        self.use_centroids = use_centroids

    def run(self) -> None:
        try:
            if speciesid is None:
                self.failed.emit("speciesid.py is missing from the app directory.")
                return
            if BioClipWorker._engine is None:
                BioClipWorker._engine = speciesid.BioCLIP()
            t0 = time.time()
            out = BioClipWorker._engine.describe_image(self.image, self.candidates,
                                                       top_k=self.top_k)
            # The personal classifier scores the SAME embedding the text ranking used.
            # Encoding the photo twice would double the cost of every identification.
            if self.personal and personal_classifier is not None:
                P, labels = self.personal
                try:
                    out["personal"] = personal_classifier.classify(
                        out["features"], P, labels, top_k=self.top_k,
                        use_centroids=self.use_centroids)
                except Exception as exc:                  # never lose the text ranking
                    out["personal_error"] = f"{type(exc).__name__}: {exc}"
            out["seconds"] = time.time() - t0
            self.done.emit(out)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Tier 2 — vision LLM over HTTP
# --------------------------------------------------------------------------- #
class VisionWorker(QThread):
    done = pyqtSignal(str, float)
    failed = pyqtSignal(str)

    def __init__(self, port: int, image: str, prompt: str, candidates: list[str],
                 ranking: list[tuple[str, float]] | None = None):
        super().__init__()
        self.port, self.image, self.prompt, self.candidates = port, image, prompt, candidates
        self.ranking = list(ranking or [])

    def run(self) -> None:
        try:
            b64 = base64.b64encode(Path(self.image).read_bytes()).decode()
            if self.ranking:
                # The classifier's ACTUAL output, with scores, and a warning that it is
                # a zero-shot image-text match rather than a determination. Feeding the
                # input list here instead was misleading: those names were only the
                # shortlist the classifier had to choose from.
                lines = [f"  {i+1}. {n} — {s*100:.1f}%" for i, (n, s) in
                         enumerate(self.ranking[:12])]
                self.prompt += (
                    "\n\nA local zero-shot classifier (BioCLIP 2) ranked these species "
                    "for this photo, most likely first:\n" + "\n".join(lines) +
                    "\n\nTreat that as a weak prior, not a determination — it scores "
                    "images against text descriptions and cannot see diagnostic detail. "
                    "Confirm or overturn it from the features you can actually see, and "
                    "say plainly if the true answer is not in its list.")
            elif self.candidates:
                # No classifier run yet: be accurate about what this list is.
                top = ", ".join(self.candidates[:40])
                self.prompt += ("\n\nFor reference, a local shortlist of species "
                                "recorded in this area was prepared, but it has NOT been "
                                "scored against this photo. If the organism is not in "
                                "it, say so — do not force a match:\n" + top)
            payload = {
                "model": "local",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "max_tokens": 700, "temperature": 0.2, "stream": False,
            }
            t0 = time.time()
            d = http_json(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                          timeout=600, payload=payload)
            self.done.emit(d["choices"][0]["message"].get("content", "").strip(),
                           time.time() - t0)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class VisionServer:
    """Owns a scratch llama-server with a vision projector.

    The model and projector are per-instance so the user can switch (see
    `choose_vision_model`); passing None falls back to the built-in defaults.
    """

    def __init__(self, port: int, model: Path | str | None = None,
                 mmproj: Path | str | None = None):
        self.port = port
        self.model = Path(model) if model else VISION_MODEL
        self.mmproj = Path(mmproj) if mmproj else VISION_MMPROJ

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url("/health"), timeout=4) as r:
                return r.status == 200
        except Exception:
            return False

    def serving_this(self) -> bool:
        """True only if the RUNNING server is already serving our model.

        A server left over from a previous selection answers /health just as
        happily, so health alone is not enough — we would silently keep using the
        old model after the user switched.
        """
        try:
            with urllib.request.urlopen(self.url("/props"), timeout=5) as r:
                return Path(json.load(r).get("model_path", "")).name == self.model.name
        except Exception:
            return False

    def files_ok(self) -> tuple[bool, list[str]]:
        missing = [str(p) for p in (LLAMA_SERVER, self.model, self.mmproj)
                   if not Path(p).is_file()]
        return (not missing), missing

    def start(self) -> tuple[bool, str]:
        if self.serving_this():
            return True, "already running"
        if self.healthy():
            self.stop()                    # a different model is loaded: swap it
        ok, missing = self.files_ok()
        if not ok:
            return False, "missing: " + ", ".join(missing)
        args = [str(LLAMA_SERVER), "-m", str(self.model),
                "--mmproj", str(self.mmproj), "--host", "127.0.0.1",
                "--port", str(self.port), "-c", "8192", "-ngl", "999"]
        logf = open(VISION_LOG, "ab")
        proc = subprocess.Popen(args, stdout=logf, stderr=logf, start_new_session=True)
        VISION_PIDFILE.write_text(str(proc.pid))
        t0 = time.time()
        while time.time() - t0 < 420:
            if self.healthy():
                return True, f"up in {time.time()-t0:.0f}s"
            if proc.poll() is not None:
                tail = ""
                try:
                    tail = VISION_LOG.read_text(errors="replace")[-800:]
                except Exception:
                    pass
                return False, f"exited ({proc.returncode})\n{tail}"
            time.sleep(1.5)
        return False, "timed out waiting for load"

    def stop(self, wait_s: int = 25) -> None:
        """Terminate the vision server and WAIT for it to actually go down.

        Fire-and-forget was misleading: the caller immediately re-checked health
        and saw the still-shutting-down process, reporting "still running".
        """
        pids: set[int] = set()
        if VISION_PIDFILE.exists():
            try:
                pids.add(int(VISION_PIDFILE.read_text().strip()))
            except Exception:
                pass
        # also catch an instance this app did not start (e.g. left by a crash)
        try:
            out = subprocess.run(["pgrep", "-f", f"llama-server.*--port {self.port}"],
                                 capture_output=True, text=True, timeout=10).stdout
            for tok in out.split():
                try:
                    pids.add(int(tok))
                except ValueError:
                    pass
        except Exception:
            pass
        for pid in pids:
            try:
                os.kill(pid, 15)
            except Exception:
                pass
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if not self.healthy():
                VISION_PIDFILE.unlink(missing_ok=True)
                return
            time.sleep(0.5)
        for pid in pids:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
        time.sleep(1.5)
        VISION_PIDFILE.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Online helper — regional/seasonal candidate list from iNaturalist
# --------------------------------------------------------------------------- #
class CandidatesWorker(QThread):
    done = pyqtSignal(list, str)
    failed = pyqtSignal(str)

    def __init__(self, lat: str, lon: str, radius_km: int, month: int | None,
                 spec: dict | None = None, place_id: int | None = None,
                 place_label: str = ""):
        super().__init__()
        self.lat, self.lon, self.radius_km, self.month = lat, lon, radius_km, month
        self.spec = spec or {}
        self.place_id = place_id
        self.place_label = place_label

    def run(self) -> None:
        try:
            names = fetch_regional_candidates(
                self.lat, self.lon, self.radius_km, self.month or None,
                spec=self.spec, place_id=self.place_id)
            if self.place_id:
                where = f"{self.place_label or self.place_id}"
            else:
                where = f"{self.lat},{self.lon} r={self.radius_km}km"
            if self.month:
                where += f" month={self.month}"
            if not names:
                self.failed.emit(f"iNaturalist returned no species for {where}.")
                return
            self.done.emit(names, f"{len(names)} species near {where}")
        except Exception as exc:
            self.failed.emit(f"iNaturalist unavailable ({type(exc).__name__}: {exc}). "
                             "Candidates unchanged — local mode still works.")


class GbifWorker(QThread):
    """Regional candidate list from GBIF occurrence records.

    Unlike the iNaturalist path this needs no API key and no account, and it is
    ranked by how often a species has been recorded at that place — so the top of
    the list is what is actually found there.
    """

    done = pyqtSignal(list, str)
    failed = pyqtSignal(str)

    def __init__(self, lat: float, lon: float, radius_km: int,
                 group: str | None, depth: str | None, limit: int = 1000):
        super().__init__()
        self.lat, self.lon, self.radius_km = lat, lon, radius_km
        self.group, self.depth, self.limit = group, depth, limit

    def run(self) -> None:
        if gbif_candidates is None:
            self.failed.emit("gbif_candidates.py is not importable.")
            return
        try:
            names, info = gbif_candidates.fetch_candidates(
                lat=self.lat, lon=self.lon, radius_km=self.radius_km,
                group=self.group, depth=self.depth, limit=self.limit)
            if not names:
                self.failed.emit(
                    f"No species recorded here — {info}.\n\n"
                    "That is usually a wrong filter or a wrong PLACE, not an empty "
                    "sea — species-rich dive sites are heavily recorded. Check the "
                    "latitude/longitude are really where the photo was taken (a "
                    "place name can resolve to a same-named town elsewhere), then "
                    "try a larger radius or a broader group.")
                return
            self.done.emit(names, info)
        except Exception as exc:
            self.failed.emit(f"GBIF unavailable ({type(exc).__name__}: {exc}). "
                             "Candidates unchanged — local mode still works.")


class PlaceWorker(QThread):
    found = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def run(self) -> None:
        try:
            places = resolve_places(self.name)
            if places:
                self.found.emit(places)
            else:
                self.failed.emit(f"No iNaturalist place matched “{self.name}”. "
                                "Try the administrative name instead — dive sites "
                                "are often filed under their municipality (e.g. the "
                                "Anilao area is “Mabini, Batangas”).")
        except Exception as exc:
            self.failed.emit(f"Place lookup unavailable ({type(exc).__name__}: {exc}). "
                             "Enter latitude/longitude manually instead.")


# --------------------------------------------------------------------------- #
# Batch mode — walk a folder, use per-photo metadata, collect results
# --------------------------------------------------------------------------- #
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


class BatchWorker(QThread):
    """Identify every photo in a folder.

    Per photo it: reads EXIF (date + GPS), optionally fills the candidate list
    from iNaturalist for that location/month, ranks with BioCLIP, and emits one
    result row. The candidate list is the accuracy lever, so a photo carrying
    GPS gets a region-specific list automatically rather than the global default.
    """

    row = pyqtSignal(dict)
    progress = pyqtSignal(int, int, str)     # done, total, current filename
    finished_ok = pyqtSignal(int, str)
    failed = pyqtSignal(str)

    def __init__(self, folder: str, recursive: bool, auto_gps: bool, online: bool,
                 fixed_candidates: list[str], radius_km: int, top_k: int,
                 month_override: int | None = None, spec: dict | None = None,
                 files: list[str] | None = None):
        super().__init__()
        self.folder = folder
        # An explicit file selection wins over the folder walk. Picking photos by hand
        # is the obvious thing to expect from "batch", and a folder is not always the
        # unit people think in.
        self.files = list(files) if files else None
        self.recursive = recursive
        self.auto_gps = auto_gps
        self.online = online
        self.fixed = fixed_candidates
        self.radius_km = radius_km
        self.top_k = top_k
        self.month_override = month_override
        self.spec = spec or {}
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _collect(self) -> list[Path]:
        if self.files:
            return [Path(f) for f in self.files if Path(f).suffix.lower() in IMG_EXT]
        root = Path(self.folder)
        it = root.rglob("*") if self.recursive else root.glob("*")
        return sorted(p for p in it
                      if p.is_file() and p.suffix.lower() in IMG_EXT)

    def run(self) -> None:  # noqa: D102
        try:
            files = self._collect()
        except Exception as exc:
            self.failed.emit(f"could not read folder: {exc}")
            return
        if not files:
            self.failed.emit(f"No images found in {self.folder}")
            return
        try:
            engine = get_engine()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        done = 0
        gps_used = 0
        for f in files:
            if self._cancel:
                break
            self.progress.emit(done, len(files), f.name)
            meta = speciesid.read_exif(f) if speciesid else {}
            cands, source = self.fixed, "list"
            if self.auto_gps and meta.get("lat") is not None:
                if self.online:
                    month = speciesid.month_of(meta) or self.month_override
                    try:
                        regional = fetch_regional_candidates(
                            meta["lat"], meta["lon"], self.radius_km, month,
                            spec=self.spec)
                        if regional:
                            # UNION, never replace. A capped regional query can
                            # omit the true species entirely — an unfiltered
                            # Sweden/June list contained no Erithacus rubecula at
                            # all, and replacing the list with it would have made
                            # the right answer unchoosable.
                            known = set(regional)
                            extra = [n for n in self.fixed if n not in known]
                            cands = regional + extra
                            source = "gps+list" if extra else "gps"
                            gps_used += 1
                    except Exception:
                        pass                 # stay offline-correct, keep the list
                else:
                    source = "list (GPS present, online off)"
            if not cands:
                self.row.emit({"photo": str(f), "name": f.name, "error": "no candidates",
                               "meta": meta})
                done += 1
                continue
            t0 = time.time()
            try:
                ranked = engine.rank(f, cands, top_k=self.top_k)
            except Exception as exc:
                self.row.emit({"photo": str(f), "name": f.name, "meta": meta,
                               "error": f"{type(exc).__name__}: {exc}"})
                done += 1
                continue
            self.row.emit({
                "photo": str(f), "name": f.name, "meta": meta,
                "candidate_source": source, "candidates": len(cands),
                "ranked": ranked, "seconds": time.time() - t0,
                "error": None,
            })
            done += 1
        msg = (f"{done} photo(s) processed · {gps_used} used a GPS-derived "
               f"candidate list" if self.online else f"{done} photo(s) processed (offline)")
        self.finished_ok.emit(done, msg)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.image_path: str | None = None
        self.last_candidates: list[str] = []
        self.last_ranking: list[tuple[str, float]] = []
        self.last_ranking_path: str | None = None
        self.vision = VisionServer(int(cfg.get("vision_port") or VISION_PORT),
                                   cfg.get("vision_model"), cfg.get("vision_mmproj"))
        self.worker = None
        self.results: list[dict] = []
        self.batch = None
        self.cand_group: str = ""      # taxon the loaded candidates belong to
        self.place_id: int | None = cfg.get("place_id") or None
        self.place_label: str = str(cfg.get("place_label") or "")
        self.placer = None
        self.setWindowTitle(APP_NAME)
        self._fit_to_screen(1400, 900)
        self.setAcceptDrops(True)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())

        m = self.menuBar().addMenu("&File")
        a = QAction("Open photo…", self); a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self.open_photo); m.addAction(a)
        b = QAction("Batch folder…", self)
        b.triggered.connect(self.run_batch); m.addAction(b)
        e1 = QAction("Export results as CSV…", self)
        e1.triggered.connect(lambda: self.export("csv")); m.addAction(e1)
        e2 = QAction("Export results as JSON…", self)
        e2.triggered.connect(lambda: self.export("json")); m.addAction(e2)
        m.addSeparator()
        q = QAction("Quit", self); q.setShortcut(QKeySequence.StandardKey.Quit)
        q.triggered.connect(self.close); m.addAction(q)

        s = self.menuBar().addMenu("&Vision server")
        a = QAction("Choose model…", self)
        a.triggered.connect(self.choose_vision_model)
        s.addAction(a)
        s.addSeparator()
        for label, fn in (("Start", self.start_vision), ("Stop", self.stop_vision),
                          ("Check", self.check_engines),
                          ("What is it serving?", self.show_serving_model)):
            act = QAction(label, self); act.triggered.connect(fn); s.addAction(act)

        c = self.menuBar().addMenu("&Your labels")
        a = QAction("Fit from labelled folder…", self)
        a.triggered.connect(self.fit_personal)
        c.addAction(a)
        a = QAction("What is fitted?", self)
        a.triggered.connect(self.show_personal)
        c.addAction(a)
        c.addSeparator()
        a = QAction("Use averages instead of nearest photo", self)
        a.setCheckable(True)
        a.setChecked(bool(self.cfg.get("personal_use_centroids", False)))
        a.toggled.connect(self._toggle_centroids)
        c.addAction(a)
        self.centroid_action = a
        a = QAction("Include in identification", self)
        a.setCheckable(True)
        a.setChecked(bool(self.cfg.get("personal_enabled", True)))
        a.toggled.connect(self._toggle_personal)
        c.addAction(a)
        self.personal_action = a
        c.addSeparator()
        a = QAction("Forget fitted classifier", self)
        a.triggered.connect(self.forget_personal)
        c.addAction(a)

        QTimer.singleShot(300, self.check_engines)

    # -- layout ------------------------------------------------------------ #
    def _build_left(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)

        row = QHBoxLayout()
        self.open_btn = QPushButton("Open photo…"); self.open_btn.clicked.connect(self.open_photo)
        row.addWidget(self.open_btn)
        self.path_lbl = QLabel("no photo  (or drag one onto the window)")
        self.path_lbl.setStyleSheet("color:#6b7280;")
        row.addWidget(self.path_lbl, 1)
        v.addLayout(row)

        self.view = QLabel("Drop a photo here")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(420, 380)
        self.view.setStyleSheet("background:#111827; color:#9ca3af; border-radius:6px;")
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.view, 1)

        # A SHORT title. A QGroupBox title does not wrap, so its width is the full
        # sentence — the previous 90-character title alone forced the window's
        # minimum width past a 1440 px laptop screen. The explanation is a wrapping
        # label inside instead, where it can reflow.
        cbox = QGroupBox("Candidate species")
        cl = QVBoxLayout(cbox)
        self.cand_hint = QLabel(
            "One scientific name per line — the classifier can only choose from this "
            "list.")
        self.cand_hint.setWordWrap(True)
        self.cand_hint.setStyleSheet("color:#6b7280;")
        cl.addWidget(self.cand_hint)
        self.cand_edit = QPlainTextEdit()
        self.cand_edit.setFixedHeight(120)
        # Deliberately starts EMPTY rather than pre-filled. A previous version
        # seeded a hardcoded European starter list here, so a dive photo was scored
        # against ten European garden insects and ten European birds without the
        # user ever choosing that. An empty box with a clear hint is honest.
        self.cand_edit.setPlaceholderText(
            "No species loaded yet. Pick a source:\n"
            "  • Marine list…  — your dive species, narrowed to a group (offline)\n"
            "  • Fill from location — what is recorded near a place + group + month\n"
            "  • Load from file… — your own list, one scientific name per line\n"
            "Or just type names here, one per line.")
        cl.addWidget(self.cand_edit)
        r2 = QHBoxLayout()
        b = QPushButton("Load from file…"); b.clicked.connect(self.load_candidates_file)
        r2.addWidget(b)
        bm = QPushButton("Marine list…")
        bm.clicked.connect(self.load_marine_group)
        bm.setToolTip("Load your own marine species list (modernised from MarineLife "
                      "Keywords 1.0), narrowed to a group — e.g. just the nudibranchs.")
        r2.addWidget(bm)
        self.gbif_btn = QPushButton("Place list (GBIF)")
        self.gbif_btn.clicked.connect(self.load_gbif)
        self.gbif_btn.setToolTip(
            "What is actually RECORDED near the latitude/longitude above, narrowed "
            "to the group you pick — ranked by how often it has been "
            "found there.\n\n"
            "This is the only source that is region-aware: your marine list is "
            "global (everything underwater, everywhere) and the iNaturalist route "
            "is an unranked summary. Use this when you know where the photo was "
            "taken.\n\n"
            "Needs the internet; no account or API key.")
        r2.addWidget(self.gbif_btn)
        cl.addLayout(r2)

        # Second row for the online switch and the live count. Keeping them in r2 made
        # that row's minimum width the SUM of six widgets, which is what pushed the
        # window past the screen edge.
        r2b = QHBoxLayout()
        self.online_chk = QCheckBox("Online (iNat, GBIF)")
        # Short label: r2b also holds the search-area and count labels, and a
        # long checkbox text was pushing the window floor past a 1260 px screen.
        self.online_chk.setToolTip(
            "Enable the network sources: iNaturalist (Find place, Fill from "
            "location) and GBIF (Place list). The marine lists and Load from "
            "file work fully offline.")
        self.online_chk.setChecked(bool(self.cfg.get("online")))
        self.online_chk.toggled.connect(self.on_online_toggle)
        r2b.addWidget(self.online_chk)
        self.add_to_list_chk = QCheckBox("add to list")
        self.add_to_list_chk.setChecked(False)
        self.add_to_list_chk.setToolTip(
            "OFF (default): loading a group REPLACES the candidate list, so choosing "
            "reef fish after nudibranchs really shows reef fish.\n\n"
            "ON: the new names are merged with whatever is already in the box. The "
            "count then covers the mixture, and the group label describes only the "
            "last load — use it when you deliberately want a combined pool.")
        r2b.addWidget(self.add_to_list_chk)
        # Persistent "where" indicator. The 4-species Anilao answer came from a place
        # lookup that silently resolved to a different Anilao 1,000 km away; the only
        # clue was the coordinates in a log line nobody reads. Showing the search area
        # permanently makes a wrong location visible BEFORE the query runs.
        self.where_lbl = QLabel()
        self.where_lbl.setStyleSheet("color:#6b7280;")
        # Ignored width policy: the text must never set the layout's minimum width.
        # A QLabel sized to its content re-inflated the window floor the moment the
        # coordinates got long (1206 -> beyond a 1260 px screen), undoing the layout
        # work — status text is exactly the kind of thing that should shrink, not push.
        self.where_lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                                     QSizePolicy.Policy.Preferred)
        self.where_lbl.setMinimumWidth(0)
        self.where_lbl.setToolTip(
            "Where the next regional lookup will search. The coordinates are the ground "
            "truth — a place NAME can match a same-named town far away (\"Anilao\" "
            "matches a town in Iloilo, ~1,000 km from the Batangas dive area).")
        r2b.addWidget(self.where_lbl)
        r2b.addStretch(1)
        # How many candidates are loaded, shown BEFORE running BioCLIP: the classifier
        # scores against exactly this many names, and the count is the cheapest way to
        # notice that a list is empty, tiny, or unexpectedly huge.
        self.cand_count = QLabel()
        self.cand_count.setStyleSheet("font-weight:600;")
        self.cand_count.setToolTip(
            "Names the classifier will score against. More names encode slower and "
            "blunt the top pick's confidence, so a regional or group-narrowed list "
            "usually beats a global one.")
        r2b.addWidget(self.cand_count)
        cl.addLayout(r2b)
        # Gated on the online switch: GBIF is a network service like iNaturalist.
        # The marine list and Load from file stay available offline.
        self._sync_online_state()
        self.cand_edit.textChanged.connect(self._update_cand_count)
        self._update_cand_count()

        rp = QHBoxLayout()
        rp.addWidget(QLabel("Place"))
        self.place = QLineEdit()
        self.place.setPlaceholderText(
            "type where you were — e.g. Bunaken, Malmö — for the species list there")
        self.place.returnPressed.connect(self.find_place)
        rp.addWidget(self.place, 1)
        self.find_btn = QPushButton("Find place")
        self.find_btn.clicked.connect(self.find_place)
        self.find_btn.setEnabled(self.online_chk.isChecked())
        rp.addWidget(self.find_btn)
        cl.addLayout(rp)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Lat"))
        self.lat = QLineEdit(str(self.cfg.get("lat", ""))); self.lat.setMaximumWidth(90)
        self.lat.textEdited.connect(self._clear_place)
        r3.addWidget(self.lat)
        r3.addWidget(QLabel("Lon"))
        self.lon = QLineEdit(str(self.cfg.get("lon", ""))); self.lon.setMaximumWidth(90)
        self.lon.textEdited.connect(self._clear_place)
        r3.addWidget(self.lon)
        r3.addWidget(QLabel("r(km)"))
        self.radius = QSpinBox(); self.radius.setRange(1, 500)
        self.radius.setValue(int(self.cfg.get("radius_km", 50))); self.radius.setMaximumWidth(80)
        r3.addWidget(self.radius)
        cl.addLayout(r3)                 # the coordinate row ends here

        # Wired HERE, after lat/lon/radius exist. Putting these connections next to the
        # where-label (built earlier) raised AttributeError: 'MainWindow' object has no
        # attribute 'lat' — the label is created before the coordinate widgets.
        # NB: the loop variable must NOT be named `w` — that is this function's own
        # widget, and rebinding it let the widget be garbage-collected mid-build
        # ("RuntimeError: wrapped C/C++ object of type QVBoxLayout has been deleted").
        for _edit in (self.lat, self.lon):
            _edit.textChanged.connect(self._update_where)
        self.radius.valueChanged.connect(self._update_where)
        self._update_where()

        # Month / group / Fill get their own row. One line holding all six controls
        # made that row's MINIMUM width the sum of them, which is what pushed the
        # window past the right edge of a 1440 px screen.
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("month (iNat)"))
        self.month = QComboBox(); self.month.addItem("any", 0)
        for i, mn in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                "Sep", "Oct", "Nov", "Dec"], 1):
            self.month.addItem(mn, i)
        # Defaults to "any", NOT the current month. Preselecting today's month looks
        # helpful but silently guts a regional query: at Anilao, nudibranchs at
        # 10-30 m are 1,357 records / 314 species with no month filter and just
        # 2 records in January. A photo's month is usually unknown anyway, so the
        # broad list is the honest default and narrowing is one click.
        self.month.setCurrentIndex(0)
        self.month.setToolTip(
            "Applies to the iNaturalist “Fill from location” list only — the GBIF "
            "Place list ignores it.\n\n"
            "Default is 'any'. Restricting to one month is expensive: GBIF coverage "
            "is sparse and survey-driven per month, and an iNaturalist query narrows "
            "the same way. Only pick a month if you know it and want that season — a "
            "month filter can turn a 300-species list into a handful.")
        r3.addWidget(self.month)
        r3.addWidget(QLabel("group"))
        self.group = QComboBox()
        for label, val in GROUPS:
            if not label:                        # section divider, not selectable
                self.group.insertSeparator(self.group.count())
            else:
                self.group.addItem(label, val)
        self.group.setCurrentIndex(0)
        self.group.setToolTip(
            "What kind of organism is it? You rarely know the species, but you "
            "usually know the group — and the group decides the candidate list.\n\n"
            "At Anilao the same location gives 120 nudibranch species "
            "(Nembrotha chamberlaini, Taringa halgerda) or 140 reef-fish species "
            "(Solenostomus paradoxus), depending on which you pick. An unfiltered "
            "list mixes land and reef life and wastes most of its 200 slots.\n\n"
            "Note: 'land · mammals' is the whole class, so it also returns whales "
            "and dolphins — use 'sea · dolphins & whales' when you mean those, and "
            "'sea · turtles' for sea turtles (they belong to Reptilia but a diver "
            "looks under sea).")
        r3.addWidget(self.group)
        self.fill_btn = QPushButton("Fill from location")
        self.fill_btn.clicked.connect(self.fill_from_location)
        self.fill_btn.setEnabled(self.online_chk.isChecked())
        r3.addWidget(self.fill_btn)
        r3.addStretch(1)
        cl.addLayout(r3)
        v.addWidget(cbox)
        return w

    def _build_right(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)

        r = QHBoxLayout()
        self.ident_btn = QPushButton("Identify (local BioCLIP)")
        self.ident_btn.setMinimumHeight(34)
        self.ident_btn.clicked.connect(self.identify)
        r.addWidget(self.ident_btn, 2)
        self.ask_btn = QPushButton("Ask vision model")
        self.ask_btn.setMinimumHeight(34)
        self.ask_btn.clicked.connect(self.ask_vision)
        r.addWidget(self.ask_btn, 1)
        v.addLayout(r)

        # Off by default: a second opinion is only worth asking for if it is
        # INDEPENDENT. Showing the model BioCLIP's list turns it into a ratifier —
        # observed in practice, where a wrong rank-1 pick was simply echoed back.
        # Tick it only when you want a cross-check against that specific shortlist.
        self.show_list_chk = QCheckBox(
            "Give the vision model the BioCLIP list (biases it)")
        self.show_list_chk.setChecked(False)
        self.show_list_chk.setToolTip(
            "OFF (default): the vision model sees only the photo, so its answer is an "
            "independent opinion — the useful thing when BioCLIP's top pick looks wrong.\n\n"
            "ON: it is also shown BioCLIP's ranked candidates with scores. That helps "
            "it land on a name it might not have recalled, but it also invites it to "
            "agree with rank 1, which defeats the point of asking twice.\n\n"
            "Either way, after it answers the app reports where its answer sat in "
            "BioCLIP's ranking — so you can see who was right, and how close.")
        v.addWidget(self.show_list_chk)

        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setFormat("idle")
        v.addWidget(self.prog)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Species", "Probability", "From"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed); self.table.setColumnWidth(0, 34)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        # Copying the name is the whole point of a result: you want to paste it into a
        # browser and check the identification yourself.
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._species_menu)
        self._copy_sc = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        self._copy_sc.activated.connect(self._copy_species)
        self.table.setToolTip(
            "Click a row and press Ctrl+C to copy the species name. "
            "Right-click for copy / open-online options.")
        v.addWidget(self.table, 2)

        tab = QTabWidget()
        self.reason = QTextBrowser()
        self.reason.setPlaceholderText(
            "Ask the vision model for diagnostic features and a second opinion.")
        # Selectable with mouse and keyboard; Ctrl+C then copies the selection. The
        # context menu adds what Ctrl+C cannot do — take the WHOLE answer in one go, and
        # follow a name mentioned in the prose out to iNaturalist/GBIF/Images — because
        # the model writes its findings as free text, not as rows.
        self.reason.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.reason.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.reason.customContextMenuRequested.connect(self._reason_menu)
        self.reason.setToolTip(
            "Select text and press Ctrl+C to copy it, or right-click to copy the whole "
            "answer / look a name up online.")
        tab.addTab(self.reason, "Vision model reasoning")

        # --- batch results + export ---
        bw = QWidget(); bl = QVBoxLayout(bw)
        self.btable = QTableWidget(0, 6)
        self.btable.setHorizontalHeaderLabels(
            ["Photo", "Taken", "GPS", "Most likely species", "Prob", "Source"])
        self.btable.verticalHeader().setVisible(False)
        self.btable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.btable.itemDoubleClicked.connect(self._open_batch_row)
        # Single-click selection previews the PHOTO in the big viewer on the left.
        # Batch rows show filenames like DSC_4821.NEF, which tell you nothing about
        # which animal you are looking at.
        self.btable.currentCellChanged.connect(self._on_batch_select)
        bh = self.btable.horizontalHeader()
        bh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4, 5):
            bh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        bl.addWidget(self.btable, 1)
        br = QHBoxLayout()
        self.batch_btn = QPushButton("Batch folder…")
        self.batch_btn.clicked.connect(self.run_batch)
        self.batch_btn.setToolTip(
            "Identify every image in a folder. Reads each photo's own GPS and date, so "
            "a set shot at one dive site gets that site's species list per photo.")
        br.addWidget(self.batch_btn)
        self.batch_files_btn = QPushButton("Batch files…")
        self.batch_files_btn.clicked.connect(self.run_batch_files)
        self.batch_files_btn.setToolTip(
            "Same as Batch folder, but you pick the photos by hand (ctrl-click for "
            "several). Use this when the shots you want are mixed in with others.")
        br.addWidget(self.batch_files_btn)
        self.batch_cancel = QPushButton("Cancel")
        self.batch_cancel.setEnabled(False)
        self.batch_cancel.clicked.connect(self._cancel_batch)
        br.addWidget(self.batch_cancel)
        self.recursive_chk = QCheckBox("subfolders")
        self.recursive_chk.setChecked(True)
        br.addWidget(self.recursive_chk)
        self.autogps_chk = QCheckBox("per-photo GPS candidates")
        self.autogps_chk.setChecked(True)
        self.autogps_chk.setToolTip(
            "If a photo carries GPS, ask iNaturalist for the species recorded near "
            "that point (and in that month) and score against those instead of the "
            "global list. Needs online features enabled; otherwise the list is used.")
        br.addWidget(self.autogps_chk)
        br.addStretch(1)
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export("csv"))
        br.addWidget(self.export_csv_btn)
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(lambda: self.export("json"))
        br.addWidget(self.export_json_btn)
        bl.addLayout(br)
        tab.addTab(bw, "Batch results")
        self.batch_tab = bw

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        tab.addTab(self.log, "Log")
        self.tabs = tab
        v.addWidget(tab, 2)
        return w

    def _fit_to_screen(self, want_w: int = 1400, want_h: int = 900,
                       avail=None) -> None:
        """Size the window to the desktop, never past it.

        `resize(1400, 900)` assumed a wide screen. On a 1440x900 laptop the window
        hung off the right edge. Ask for the requested size but clamp it to what the
        screen actually offers; `avail` is injectable so the clamp can be tested
        without a real desktop.
        """
        if avail is None:
            try:
                scr = self.screen() or QApplication.primaryScreen()
                avail = scr.availableGeometry() if scr else None
            except Exception:
                avail = None
        if avail is None or avail.width() <= 0:
            self.resize(want_w, want_h)
            return
        # Hard ceiling: nothing may push the window past the desktop. Any child whose
        # minimum width grows with its content — a button that embeds its state in its
        # own label did exactly that, +92 px — otherwise drags the window off-screen,
        # because Qt honours a minimum size over the requested size.
        self.setMaximumSize(avail.width(), avail.height())
        w = min(want_w, max(640, avail.width() - 40))
        h = min(want_h, max(480, avail.height() - 80))
        self.resize(w, h)
        self.move(avail.x() + 10, avail.y() + 10)

    # -- thread lifetime ---------------------------------------------------- #
    def _reap(self) -> None:
        """Drop a worker reference only AFTER its run() has returned.

        Never null the reference from a slot connected to a signal the worker
        emits *during* run(): dropping the last reference destroys the QThread
        mid-execution and aborts the process ("QThread: Destroyed while thread
        is still running"). QThread.finished fires after run() returns, so it
        is the only safe place to clear it.
        """
        w = self.sender()
        if w is self.worker:
            self.worker = None
        if w is getattr(self, "api", None):
            self.api = None
        if w is getattr(self, "batch", None):
            self.batch = None
        if w is getattr(self, "placer", None):
            self.placer = None
        if w is getattr(self, "gbiFer", None):
            self.gbiFer = None
        if w is not None:
            w.deleteLater()

    def _track(self, worker) -> None:
        worker.finished.connect(self._reap)

    # -- engines ----------------------------------------------------------- #
    def check_engines(self) -> None:
        bits = []
        if speciesid:
            st = speciesid.BioCLIP().status()
            bits.append(f"BioCLIP {'✓' if st['ready'] else '✗ missing'}")
        else:
            bits.append("BioCLIP ✗ (speciesid.py)")
        ok, missing = self.vision.files_ok()
        if not ok:
            bits.append("vision ✗ files")
        else:
            # name the model: previously which model served "Ask vision model" was
            # completely invisible in the UI.
            state = "✓" if self.vision.healthy() else "○"
            bits.append(f"vision {state} {self.vision.model.stem[:30]}")
        bits.append("online ON" if self.online_chk.isChecked() else "offline")
        # Show where the candidate list would come from, so a stale location
        # cannot sit there unnoticed.
        lat, lon = self.lat.text().strip(), self.lon.text().strip()
        if self.place_label or (lat and lon):
            bits.append(f"place: {self.place_label or f'{lat},{lon}'}")
        else:
            bits.append("place: none set")
        self.statusBar().showMessage("   |   ".join(bits), 20000)

    # -- vision model selection -------------------------------------------- #
    def choose_vision_model(self) -> None:
        """Pick which vision model answers "Ask vision model".

        Discovery reads each GGUF's metadata rather than trusting filenames, since
        a file named "Qwen-Image-2.1-Turbo-Abliterated-Uncensored" is in fact a
        Qwen3-VL-8B vision model.
        """
        if vision_models is None:
            QMessageBox.warning(self, APP_NAME,
                                "vision_models.py is missing from the app directory.")
            return
        self.prog.setFormat("scanning ~/Models for vision models…")
        QApplication.processEvents()
        try:
            found = vision_models.discover([Path.home() / "Models"])
        except Exception as exc:
            self.prog.setFormat("idle")
            QMessageBox.warning(self, APP_NAME, f"Could not scan for models:\n{exc}")
            return
        self.prog.setFormat("idle")
        opts = [o for o in found if o.vision_ok and o.projector]
        if not opts:
            QMessageBox.information(
                self, APP_NAME,
                "No vision-capable GGUF found under ~/Models.\n\n"
                "A vision model needs its own mmproj projector file as well as "
                "the model itself.")
            return
        items = [f"{o.name or o.model.stem}   ({o.size_gb:.1f} GB · {o.arch})"
                 for o in opts]
        cur = next((i for i, o in enumerate(opts) if o.model == self.vision.model), 0)
        choice, ok = QInputDialog.getItem(
            self, "Vision model",
            "Which model should answer “Ask vision model”?\n\n"
            "The identification itself always comes from BioCLIP — this only sets "
            "the model used for the written reasoning / second opinion.",
            items, cur, False)
        if not ok or not choice:
            return
        o = opts[items.index(choice)]
        self.cfg["vision_model"] = str(o.model)
        self.cfg["vision_mmproj"] = str(o.projector)
        save_cfg(self.cfg)
        was_running = self.vision.healthy()
        self.vision = VisionServer(int(self.cfg.get("vision_port") or VISION_PORT),
                                   o.model, o.projector)
        self.log.appendPlainText(
            f"vision model set to {o.model.name}\n     {o.model}\n     + {o.projector}")
        if was_running:
            self.log.appendPlainText("restarting the vision server on the new model…")
            self.start_vision()
        else:
            self.check_engines()

    def show_serving_model(self) -> None:
        """Report exactly what the running server has loaded (ground truth)."""
        if not self.vision.healthy():
            QMessageBox.information(self, APP_NAME,
                                    "The vision server is not running.\n\n"
                                    f"Configured model:\n{self.vision.model}")
            return
        try:
            with urllib.request.urlopen(self.vision.url("/props"), timeout=8) as r:
                d = json.load(r)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not query the server:\n{exc}")
            return
        mods = d.get("modalities") or {}
        QMessageBox.information(
            self, APP_NAME,
            f"Serving:\n{d.get('model_path')}\n\n"
            f"modalities: {mods}\n"
            f"context: {d.get('default_generation_settings', {}).get('n_ctx')}\n"
            f"build: {d.get('build_info')}")

    def start_vision(self) -> None:
        self.prog.setFormat("starting vision server…")
        ok, msg = self.vision.start()
        self.log.appendPlainText(f"vision server start: {msg}")
        self.prog.setFormat("idle" if not ok else "vision ready")
        if not ok:
            QMessageBox.warning(self, APP_NAME, f"Could not start the vision server:\n{msg}")
        self.check_engines()

    def stop_vision(self) -> None:
        self.vision.stop()
        self.log.appendPlainText("vision server stopped")
        self.check_engines()

    # -- photo ------------------------------------------------------------- #
    def open_photo(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open photo",
                                           self.cfg.get("last_dir", str(Path.home())),
                                           "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if p:
            self.load_photo(p)

    def load_photo(self, path: str) -> None:
        self.image_path = path
        self.cfg["last_dir"] = str(Path(path).parent)
        save_cfg(self.cfg)
        pm = QPixmap(path)
        if pm.isNull():
            QMessageBox.warning(self, APP_NAME, f"Could not read image:\n{path}")
            return
        self.view.setPixmap(pm.scaled(self.view.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        self.path_lbl.setText(f"{Path(path).name}   ({pm.width()}×{pm.height()})")
        self._apply_photo_metadata(path)
        self.statusBar().showMessage(f"loaded {path}", 10000)

    def _apply_photo_metadata(self, path: str) -> None:
        """Use the photo's own EXIF/GPS to fill what it can.

        Batch mode has always read EXIF, but single-photo mode ignored it entirely —
        so opening a GPS-tagged photo left Lat/Lon empty even though the answer was
        in the file. The date is reported rather than applied as a filter: a month
        filter cuts a regional list drastically (314 species -> 2 records at Anilao),
        so it stays an explicit choice.
        """
        if not speciesid:
            return
        try:
            meta = speciesid.read_exif(path)
        except Exception as exc:
            self.log.appendPlainText(f"photo metadata: could not read ({exc})")
            return
        if not meta.get("has_exif"):
            self.log.appendPlainText(
                "photo metadata: none found — many images are re-encoded or stripped, "
                "so there is no GPS or date to use")
            return

        bits = []
        if meta.get("camera"):
            bits.append(meta["camera"])
        if meta.get("datetime"):
            bits.append(f"taken {meta['datetime'].replace('T', ' ')}")
        has_gps = meta.get("lat") is not None and meta.get("lon") is not None
        if has_gps:
            bits.append(f"GPS {meta['lat']}, {meta['lon']}")
        if bits:
            self.log.appendPlainText("photo metadata: " + " | ".join(bits))

        if has_gps:
            self.lat.setText(str(meta["lat"]))
            self.lon.setText(str(meta["lon"]))
            # The coordinates now come from the photo, not from a place lookup, so
            # drop any place id or it would keep driving the iNaturalist query.
            self.place_id, self.place_label = None, ""
            self._update_where()
            self.statusBar().showMessage(
                f"filled Lat/Lon from the photo's GPS ({meta['lat']}, {meta['lon']}) "
                "— press “Place list (GBIF)” for what is recorded there", 20000)
        else:
            # Log it, do not only flash it in the status bar: a status message is gone
            # in seconds, and "why are Lat/Lon still empty?" is the obvious question.
            stamp = (meta.get("datetime") or "").replace("T", " ")
            self.log.appendPlainText(
                "photo metadata: no GPS in this photo"
                + (f" (taken {stamp})" if stamp else "")
                + " — set the location by hand for a regional list")
            if stamp:
                self.statusBar().showMessage(
                    f"no GPS in the photo; taken {stamp} — set the location by hand "
                    "for a regional list", 20000)

    # drag & drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p:
                self.load_photo(p)
                break

    # -- candidates -------------------------------------------------------- #
    def _update_where(self) -> None:
        """Show the area the next regional lookup will search, always visible."""
        try:
            lat = float(self.lat.text().strip())
            lon = float(self.lon.text().strip())
        except (TypeError, ValueError):
            self.where_lbl.setText("no area set")
            self.where_lbl.setStyleSheet("color:#b91c1c;")
            return
        # Compact on purpose: the full sentence lives in the tooltip, because the
        # visible text must stay short enough not to widen the window.
        place = f" {self.place_label}" if getattr(self, "place_label", "") else ""
        self.where_lbl.setText(f"@ {lat:.3f}, {lon:.3f}{place} ±{self.radius.value()} km")
        self.where_lbl.setStyleSheet("color:#6b7280;")

    def _update_cand_count(self) -> None:
        """Show how many candidate names are loaded, before anything is scored.

        Counts via candidates() so it reflects what the classifier will actually use
        (blank lines ignored), not the raw line count of the box. The group is shown
        too: with a group selected every name is guaranteed to be in that taxon, and
        saying so removes the doubt about what the list actually contains.
        """
        try:
            n = len(self.candidates())
        except Exception:
            n = 0
        grp = getattr(self, "cand_group", "")
        suffix = f" — {grp}" if grp else ""
        if n == 0:
            self.cand_count.setText("0 candidates")
            self.cand_count.setStyleSheet("color:#b91c1c; font-weight:600;")
        else:
            self.cand_count.setText(
                f"{n:,} candidate{'s' if n != 1 else ''}{suffix}")
            self.cand_count.setStyleSheet("font-weight:600;")

    def candidates(self) -> list[str]:
        return [l.strip() for l in self.cand_edit.toPlainText().splitlines()
                if l.strip() and not l.startswith("#")]

    def load_candidates_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Candidate list", str(Path.home()),
            "Species list (*.txt *.csv *.tsv);;Text (*.txt);;CSV (*.csv);;All files (*)")
        if not p:
            return
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not read {p}:\n{exc}")
            return
        names = speciesid.parse_candidate_file(text) if speciesid else \
            [l.strip() for l in text.splitlines() if l.strip()]
        if not names:
            QMessageBox.information(self, APP_NAME,
                                    f"No species names found in {Path(p).name}.")
            return
        self.cand_group = ""
        keep = [n for n in self.candidates() if n not in set(names)]
        self.cand_edit.setPlainText("\n".join(names + keep))
        self._update_cand_count()
        self.log.appendPlainText(
            f"loaded {len(names)} candidate(s) from {p}"
            + (f"; kept {len(keep)} already in the list" if keep else ""))

    def load_marine_group(self) -> None:
        """Load a group from the user's own marine list, unioned into the box.

        Narrowing is the point: a diver who knows it is a nudibranch should load
        ~880 names rather than ~7,200. Never replaces what is already in the box.
        """
        if not MARINE_LISTS.is_dir():
            QMessageBox.warning(
                self, APP_NAME,
                f"No marine list found at:\n{MARINE_LISTS}\n\n"
                "Create it with modernize_marinelife.py then "
                "build_mkw_candidates.py.")
            return
        entries: list[tuple[str, Path]] = []
        for label, fn in MARINE_GROUPS:
            p = MARINE_LISTS / fn
            if not p.is_file():
                continue
            try:
                n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
            except Exception:
                n = 0
            entries.append((f"{label}   ({n:,} names)", p))
        if not entries:
            QMessageBox.information(self, APP_NAME,
                                    f"{MARINE_LISTS} exists but holds no group files.")
            return
        items = [e[0] for e in entries]
        choice, ok = QInputDialog.getItem(
            self, "Marine species list",
            "Which group?\n\nLoading everything (~7,200 names) is slower to encode "
            "and blunts the top pick's confidence — narrowing to what you actually "
            "saw keeps it sharp.",
            items, 0, False)
        if not ok or not choice:
            return
        path = entries[items.index(choice)][1]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not read {path.name}:\n{exc}")
            return
        names = (speciesid.parse_candidate_file(text) if speciesid
                 else [l.strip() for l in text.splitlines() if l.strip()])
        if not names:
            QMessageBox.information(self, APP_NAME, f"{path.name} holds no names.")
            return
        self.cand_group = path.stem.replace("_", " ")
        replace = not getattr(self, "add_to_list_chk", None) or \
            not self.add_to_list_chk.isChecked()
        kept = self._set_candidates(names, replace=replace, source="marine list")
        self.log.appendPlainText(
            f"marine list: {path.stem} — {len(names):,} names"
            + ("" if replace else f"; kept {kept} already in the list"))
        self.statusBar().showMessage(
            f"loaded {len(names):,} marine species from {path.name}", 20000)

    def on_online_toggle(self, on: bool) -> None:
        self.cfg["online"] = bool(on); save_cfg(self.cfg)
        self.fill_btn.setEnabled(on)
        self.find_btn.setEnabled(on)
        self._sync_online_state()
        self.check_engines()

    # -- GBIF: what is actually recorded at a place -------------------------- #
    def load_gbif(self) -> None:
        """Build a regional, ranked candidate list from GBIF occurrence records.

        Ranked by how often each species has been recorded there, so the order is
        meaningful. Unioned into the box like every other source — never replaces.
        """
        if gbif_candidates is None:
            QMessageBox.warning(self, APP_NAME,
                                "gbif_candidates.py is not importable.")
            return
        try:
            lat = float(self.lat.text().strip())
            lon = float(self.lon.text().strip())
        except ValueError:
            QMessageBox.information(
                self, APP_NAME,
                "Set Latitude and Longitude first — GBIF builds the list from the "
                "records around that point.\n\n"
                "If you don't know the coordinates, use “Find place” above, or read "
                "them off a photo that has GPS (Open photo…).")
            return
        group = self.group.currentData()
        # No depth argument: the depth selector was removed because GBIF's depth field
        # is populated inconsistently, and filtering on it silently drops real species
        # (143 of 541 nudibranch species at Anilao have no depth recorded at all). The
        # group filter does the useful narrowing.
        # Month is deliberately NOT passed: GBIF's per-month coverage is survey-driven
        # and sparse, and a month filter silently reduced this same query from 314
        # species to 2. The combo still drives the iNaturalist location list.
        self.gbif_btn.setEnabled(False)
        self.gbif_btn.setText("Asking GBIF…")
        self.gbiFer = GbifWorker(lat, lon, self.radius.value(), group, None)
        self.gbiFer.done.connect(self._on_gbif)
        self.gbiFer.failed.connect(self._on_gbif_fail)
        self._track(self.gbiFer)
        self.gbiFer.start()

    def _set_candidates(self, names: list[str], *, replace: bool,
                        source: str) -> int:
        """Put a new candidate list in the box. Returns how many were kept.

        `replace` defaults to True for the place/group sources: choosing reef fish and
        pressing the button means "show me reef fish", not "add reef fish to the 541
        nudibranchs still in the box". The old union behaviour made the count grow
        monotonically (541 nudibranchs -> 1,211 mixed) while the label claimed the new
        group — which is what made it look broken.
        """
        if replace:
            self.cand_edit.setPlainText("\n".join(names))
            self._update_cand_count()
            return 0
        known = set(names)
        keep = [n for n in self.candidates() if n not in known]
        self.cand_edit.setPlainText("\n".join(names + keep))
        self._update_cand_count()
        return len(keep)

    def _sync_online_state(self) -> None:
        """Enable the network buttons and SAY why when they are off.

        A disabled button that looks identical to an enabled one is what produced
        "I click on GBIF but it says iNat, GBIF Online" — the button was inert and the
        only clue was a checkbox label. Say why, but NOT in the button text: the text
        "Place list (GBIF) — needs Online" is 193 px against 101 px for the plain
        label, and that 92 px swing inflated the button row, the left pane and the
        whole window's minimum width every time Online was toggled — which is exactly
        when the window was seen to grow past the screen edge. Disabled state plus a
        tooltip carries the same information at a constant width.
        """
        on = self.online_chk.isChecked()
        self.gbif_btn.setEnabled(on)
        self.gbif_btn.setText("Place list (GBIF)")
        self.gbif_btn.setToolTip(
            "Species recorded near the search area, from GBIF. Needs “Online features”."
            if not on else
            "Species recorded near the search area, from GBIF occurrences.")
        if hasattr(self, "find_btn"):
            self.find_btn.setEnabled(on)

    def _group_label_for_vision(self) -> str:
        """Clean, human label for the taxon currently selected, for the count."""
        try:
            label = self.group.currentText()
        except Exception:
            return ""
        if not label or label.lower().startswith("no group"):
            return ""
        short = label.split("·")[-1].strip() or label
        taxon = None
        if gbif_candidates is not None:
            gname = gbif_candidates.group_from_spec(self.group.currentData())
            if gname:
                taxon = gbif_candidates.GROUP_TAXA.get(gname, (None, None))[0]
        # bony fishes have no GBIF key, so say that rather than pretending.
        if taxon:
            return taxon
        return f"{short} (GBIF can't filter this group)"

    def _on_gbif(self, names: list[str], info: str) -> None:
        self.gbif_btn.setEnabled(self.online_chk.isChecked())
        self.gbif_btn.setText("Place list (GBIF)")
        # Label from the GROUP SELECTOR, not scraped out of the provenance string: the
        # regex used to yield nonsense like "filter — bony_fishes (unfiltered)".
        self.cand_group = self._group_label_for_vision()
        replace = not getattr(self, "add_to_list_chk", None) or \
            not self.add_to_list_chk.isChecked()
        kept = self._set_candidates(names, replace=replace, source="GBIF")
        hint = ""
        if len(names) < 8:
            hint = ("  — only a few species came back. That usually means the SEARCH "
                    "AREA is not where you think (a place name can resolve to a "
                    "same-named town far away — check the coordinates shown above), or "
                    "the radius is too tight.")
        if replace:
            self.log.appendPlainText(f"GBIF: {info}" + hint)
        else:
            self.log.appendPlainText(
                f"GBIF: {info}; kept {kept} already loaded (add mode)" + hint)
        self.statusBar().showMessage(
            f"GBIF loaded {len(names):,} recorded species (most-recorded first)"
            + hint, 20000)

    def _on_gbif_fail(self, msg: str) -> None:
        self.gbif_btn.setEnabled(self.online_chk.isChecked())
        self.gbif_btn.setText("Place list (GBIF)")
        self.log.appendPlainText(f"GBIF: {msg}")
        self.statusBar().showMessage(msg, 15000)

    # -- place-name lookup (for photos that carry no GPS at all) ------------ #
    def find_place(self) -> None:
        name = self.place.text().strip()
        if not name:
            QMessageBox.information(self, APP_NAME, "Type a place name first.")
            return
        self.find_btn.setEnabled(False)
        self.prog.setFormat(f"looking up “{name}”…")
        self.placer = PlaceWorker(name)
        self.placer.found.connect(self._on_places)
        self.placer.failed.connect(self._on_place_fail)
        self._track(self.placer)
        self.placer.start()

    def _on_places(self, places: list[dict]) -> None:
        self.find_btn.setEnabled(True)
        self.prog.setFormat("idle")
        if len(places) == 1:
            pick = places[0]
            # A single match is auto-accepted, so ASK. "Anilao" matches exactly one
            # iNaturalist place — Anilao, ILOILO, ~1,000 km from the Batangas dive
            # area — which returns 4 nudibranch species instead of ~390. A log note
            # was not enough: this is silently wrong and looks like a broken query.
            ans = QMessageBox.question(
                self, "Check these coordinates",
                f"Only one place matched: {pick.get('name')}\n"
                f"at {pick.get('lat')}, {pick.get('lon')}\n\n"
                "Place names are ambiguous — the same name can be a town a thousand "
                "kilometres away, and that silently gives a wrong species list.\n\n"
                "Is this where the photo was actually taken?\n\n"
                "If not, choose No and enter the coordinates by hand, or search the "
                "municipality instead (the Anilao dive area is “Mabini, Batangas”, "
                "about 13.715, 120.896).",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                self.log.appendPlainText(
                    f"place lookup: rejected {pick.get('name')} at "
                    f"{pick.get('lat')}, {pick.get('lon')} — coordinates unchanged. "
                    "Try the municipality name, or type lat/lon directly.")
                self.statusBar().showMessage("place not applied — coordinates unchanged",
                                             15000)
                return
            self.log.appendPlainText(
                f"note: only one place matched, accepted {pick.get('name')} at "
                f"{pick.get('lat')}, {pick.get('lon')} after confirmation")
        else:
            items = [f"{p['name']}    ({p['lat']}, {p['lon']})" for p in places]
            choice, ok = QInputDialog.getItem(
                self, "Which place?",
                f"{len(places)} places match that name.\n\nPick the right one and check "
                "the coordinates — the wrong region gives a confident wrong species "
                "list.",
                items, 0, False)
            if not ok or not choice:
                self.log.appendPlainText("place lookup: cancelled")
                return
            pick = places[items.index(choice)]
        self.place_id = pick.get("id")
        self.place_label = str(pick.get("name") or "")
        if pick.get("lat") is not None:
            self.lat.setText(str(pick["lat"]))
            self.lon.setText(str(pick["lon"]))
        self.log.appendPlainText(
            f"place: {self.place_label} (id={self.place_id}), centre "
            f"{pick.get('lat')},{pick.get('lon')} — now press “Place list (GBIF)” "
            "for what is recorded there, or “Fill from location” for iNaturalist")
        self._update_where()
        self.statusBar().showMessage(
            f"resolved to {self.place_label} (id {self.place_id}) — check the "
            f"coordinates are the right place", 20000)

    def _on_place_fail(self, msg: str) -> None:
        self.find_btn.setEnabled(True)
        self.prog.setFormat("idle")
        self.log.appendPlainText("place: " + msg)
        self.statusBar().showMessage(msg[:160], 15000)

    def _clear_place(self) -> None:
        """Editing lat/lon by hand means the user no longer means the found place."""
        if self.place_id:
            self.place_id, self.place_label = None, ""
            self._update_where()

    def fill_from_location(self) -> None:
        has_coords = bool(self.lat.text().strip() and self.lon.text().strip())
        if not (has_coords or self.place_id):
            QMessageBox.information(
                self, APP_NAME,
                "Type a place name and press “Find place”, or enter a latitude and "
                "longitude.\n\n(Most photos carry no GPS, so the place has to come "
                "from you.)")
            return
        self.cfg.update(lat=self.lat.text().strip(), lon=self.lon.text().strip(),
                        radius_km=self.radius.value(),
                        place_id=self.place_id, place_label=self.place_label)
        save_cfg(self.cfg)
        self.fill_btn.setEnabled(False)
        self.prog.setFormat("asking iNaturalist…")
        m = self.month.currentData() or None
        self.api = CandidatesWorker(self.lat.text().strip(), self.lon.text().strip(),
                                    self.radius.value(), m,
                                    spec=self.group.currentData(),
                                    place_id=self.place_id, place_label=self.place_label)
        self.api.done.connect(self._on_candidates)
        self.api.failed.connect(self._on_candidates_fail)
        self._track(self.api)
        self.api.start()

    def _on_candidates(self, names: list[str], info: str) -> None:
        self.fill_btn.setEnabled(True)
        self.prog.setFormat("idle")
        # UNION, never replace — same rule as batch mode. A capped regional query
        # can omit the true species entirely (an unfiltered Sweden/June list had
        # no Erithacus rubecula at all), and replacing the list would make the
        # right answer unchoosable.
        known = set(names)
        extra = [n for n in self.candidates() if n not in known]
        self.cand_edit.setPlainText("\n".join(names + extra))
        self.log.appendPlainText(
            f"iNaturalist: {info} — added {len(names)} regional species, kept "
            f"{len(extra)} of yours ({len(names)+len(extra)} total)")

    def _on_candidates_fail(self, msg: str) -> None:
        self.fill_btn.setEnabled(True)
        self.prog.setFormat("idle")
        self.log.appendPlainText("iNaturalist: " + msg)
        self.statusBar().showMessage(msg.replace("\n", " ")[:160], 15000)

    # -- GBIF, in the app -------------------------------------------------- #
    def lookup_gbif(self, name: str) -> None:
        """Show what GBIF's backbone says about a name, without opening gbif.org.

        The website is unusable for this on many networks: its Cloudflare layer answers
        403 to every /species/… deep link while the homepage loads. The API answers fine,
        so the verdict is fetched and displayed here — and it is the more useful answer
        anyway, because it says whether the name is ACCEPTED, a SYNONYM, or unknown.
        """
        if gbif_candidates is None:
            QMessageBox.warning(self, APP_NAME, "gbif_candidates.py is missing.")
            return
        if getattr(self, "gbif_worker", None):
            return
        self.log.appendPlainText(f"GBIF lookup: {name}  (api.gbif.org — the website "
                                 f"refuses deep links on some networks)")
        self.gbif_worker = GbifLookupWorker(name)
        self.gbif_worker.done.connect(self._show_gbif)
        self.gbif_worker.failed.connect(self._gbif_failed)
        self._track(self.gbif_worker)
        self.gbif_worker.start()

    def _gbif_failed(self, msg: str) -> None:
        self.gbif_worker = None
        self.log.appendPlainText(f"GBIF lookup failed: {msg}")
        QMessageBox.warning(self, APP_NAME, f"GBIF lookup failed.\n\n{msg}")

    def _show_gbif(self, d: dict) -> None:
        self.gbif_worker = None
        mt = d.get("matchType") or "NONE"
        name = d.get("query") or "?"
        if mt == "NONE":
            self.log.appendPlainText(f"GBIF: no match for “{name}”.")
            QMessageBox.information(
                self, APP_NAME,
                f"GBIF does not know “{name}”.\n\n"
                f"That usually means a misspelling, a name GBIF has not indexed, or a "
                f"trade/common name rather than a scientific one.")
            return
        lines = [f"{name}", ""]
        lines.append(f"match:      {mt}")
        if d.get("status"):
            lines.append(f"status:     {d['status']}")
        if d.get("rank"):
            lines.append(f"rank:       {d['rank']}")
        if d.get("scientificName"):
            lines.append(f"GBIF name:  {d['scientificName']}")
        if d.get("accepted") and d.get("accepted") != d.get("scientificName"):
            lines.append(f"accepted as: {d['accepted']}")
        path = " › ".join(x for x in (d.get("kingdom"), d.get("phylum"), d.get("class"),
                                      d.get("order"), d.get("family"), d.get("genus")) if x)
        if path:
            lines.append(f"lineage:    {path}")
        if d.get("records") is not None:
            lines.append(f"records:    {d['records']:,} worldwide")
        if mt != "EXACT":
            lines.append("")
            lines.append("Not an EXACT match — GBIF is offering something close, so check "
                         "the name rather than trusting it.")
        text = "\n".join(lines)
        self.log.appendPlainText("GBIF: " + text.replace("\n", " | "))
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(text)
        box.setInformativeText("In-app lookup: gbif.org itself refuses deep links from "
                               "some networks (Cloudflare 403).")
        worms = box.addButton("Open in WoRMS", QMessageBox.ButtonRole.ActionRole)
        inat = box.addButton("Open in iNaturalist", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is worms:
            QDesktopServices.openUrl(QUrl(
                "https://www.marinespecies.org/aphia.php?p=taxlist&searchpar="
                + name.replace(" ", "+")))
        elif box.clickedButton() is inat:
            QDesktopServices.openUrl(QUrl(
                f"https://www.inaturalist.org/search?q={name}"))

    # -- your own labels --------------------------------------------------- #
    def _personal_set(self):
        """(prototypes, labels) when fitted and enabled, else None.

        Cached against the file's mtime so refitting from the menu takes effect at once
        without re-reading a multi-megabyte archive on every identification.
        """
        if not bool(self.cfg.get("personal_enabled", True)) or personal_classifier is None:
            return None
        p = personal_path()
        try:
            st = p.stat()
        except OSError:
            self._personal_cache = (None, None)
            return None
        key = (str(p), st.st_mtime, st.st_size)
        cache = getattr(self, "_personal_cache", None)
        if cache and cache[0] == key:
            return cache[1]
        loaded = personal_classifier.load(p)
        val = (loaded[0], loaded[1]) if loaded else None
        self._personal_cache = (key, val)
        return val

    def fit_personal(self) -> None:
        start = str(self.cfg.get("last_dir") or (Path.home() / "Pictures"))
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of labelled photos — the filename is the label", start)
        if not folder:
            return
        if getattr(self, "personal_worker", None):
            return
        self.prog.setValue(0)
        self.prog.setFormat("fitting from your labels…")
        self.log.appendPlainText(
            f"Fitting personal classifier from {folder}\n"
            "  each filename must start with the species, e.g. Chromodoris_anna_01.jpg")
        self.personal_worker = PersonalFitWorker(folder, str(personal_path()))
        self.personal_worker.done.connect(self._on_fit)
        self.personal_worker.failed.connect(self._on_fit_fail)
        self._track(self.personal_worker)
        self.personal_worker.start()

    def _on_fit(self, meta: dict) -> None:
        self.personal_worker = None
        self.prog.setValue(100)
        self.prog.setFormat(f"fitted — {meta.get('species')} species")
        self._personal_cache = (None, None)          # force a reload on next identify
        counts = meta.get("examples_per_species", {})
        single = sum(1 for v in counts.values() if v == 1)
        self.log.appendPlainText(
            f"Personal classifier fitted: {meta.get('photos')} photos, "
            f"{meta.get('species')} species ({single} with a single example).\n"
            f"  saved to {personal_path()}\n"
            f"  identification now answers from your labels first, and falls back to the "
            f"candidate-list text ranking for anything you have not labelled.")
        if meta.get("unparsed_files"):
            self.log.appendPlainText(
                f"  {meta['unparsed_files']} files were ignored — the filename did not "
                f"start with a species name (camera names like OI3A2349.jpg cannot be "
                f"labels).")
        self.statusBar().showMessage(
            f"Fitted from your labels: {meta.get('species')} species", 20000)

    def _on_fit_fail(self, msg: str) -> None:
        self.personal_worker = None
        self.prog.setFormat("fit failed")
        self.log.appendPlainText(f"Fitting failed: {msg}")
        QMessageBox.warning(self, APP_NAME, f"Could not fit from that folder.\n\n{msg}")

    def show_personal(self) -> None:
        loaded = personal_classifier.load(personal_path()) if personal_classifier else None
        if not loaded:
            QMessageBox.information(
                self, APP_NAME,
                "Nothing fitted yet.\n\n"
                "Rename a folder of your own photos so each filename starts with the "
                "species — Chromodoris_anna_01.jpg, Chromodoris_anna_02.jpg — then use "
                "Your labels ▸ Fit from labelled folder…\n\n"
                "The fitted file holds embeddings of your photographs and stays in your "
                "config directory; it is never written into this app or a repository.")
            return
        _, labels, meta = loaded
        counts = meta.get("examples_per_species", {})
        single = sum(1 for v in counts.values() if v == 1)
        QMessageBox.information(
            self, APP_NAME,
            f"Fitted from {meta.get('photos')} photos of {meta.get('species')} species.\n"
            f"{single} species have only one example — fine, but a second photo of each "
            f"raises the ceiling.\n\n"
            f"Source folder: {meta.get('source_folder')}\n"
            f"Stored at: {personal_path()}\n\n"
            f"Matching: {'species averages' if self.cfg.get('personal_use_centroids') else 'nearest single photo (1-NN)'}\n"
            f"Confidence bands: confident at margin ≥ "
            f"{meta.get('confident_margin', 0.06)}, uncertain below that.\n\n"
            f"Measured on held-out photos of species with labels: 90.0% top-1, against "
            f"65.0% for the text ranking alone. Species with no label cannot be answered "
            f"this way, so the text ranking still runs beside it.")

    def forget_personal(self) -> None:
        p = personal_path()
        if not p.exists():
            QMessageBox.information(self, APP_NAME, "Nothing is fitted.")
            return
        if QMessageBox.question(
                self, APP_NAME,
                f"Delete the fitted classifier?\n\n{p}\n\n"
                "Your photographs are not touched — only the embeddings fitted from them.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            p.unlink()
            side = p.with_suffix(p.suffix + ".json")
            if side.exists():
                side.unlink()
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not delete it: {exc}")
            return
        self._personal_cache = (None, None)
        self.log.appendPlainText("Fitted classifier deleted; identification is text-only again.")

    def _toggle_centroids(self, on: bool) -> None:
        self.cfg["personal_use_centroids"] = bool(on)
        save_cfg(self.cfg)

    def _toggle_personal(self, on: bool) -> None:
        self.cfg["personal_enabled"] = bool(on)
        save_cfg(self.cfg)

    # -- actions ----------------------------------------------------------- #
    def identify(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, APP_NAME, "Open a photo first.")
            return
        cands = self.candidates()
        if not cands:
            QMessageBox.information(
                self, APP_NAME,
                "No candidate species loaded.\n\n"
                "BioCLIP can only choose from a list you give it — the list IS the "
                "answer space. Load one with:\n\n"
                "  •  Marine list…  — your dive species, narrowed to a group\n"
                "  •  Fill from location  — what's recorded near a place + group\n"
                "  •  Load from file…  — your own list, one name per line\n\n"
                "or type names into the box.")
            return
        if self.worker:
            return
        self.ident_btn.setEnabled(False)
        self.prog.setValue(0)
        self.prog.setFormat(f"BioCLIP: scoring {len(cands)} candidates…")
        self.last_candidates = cands[:40]
        personal = self._personal_set()
        self.worker = BioClipWorker(
            self.image_path, cands, int(self.cfg.get("top_k", 15)),
            personal=personal,
            use_centroids=bool(self.cfg.get("personal_use_centroids", False)))
        if personal is None and bool(self.cfg.get("personal_enabled", True)) \
                and personal_classifier is not None:
            self.log.appendPlainText(
                "No personal classifier fitted — ranking from the candidate list only. "
                "Your labels ▸ Fit from labelled folder… makes it answer from your own "
                "photos first.")
        self.worker.done.connect(self._on_bio)
        self.worker.failed.connect(self._on_fail)
        self._track(self.worker)
        self.worker.start()

    def _on_bio(self, out: dict) -> None:
        self.ident_btn.setEnabled(True)
        self.prog.setValue(100)
        self.prog.setFormat(f"done — {out.get('seconds',0):.1f}s")
        ranked = out.get("ranked", [])
        personal = out.get("personal")
        # Labels first, text ranking filling the rest — each row says which produced it,
        # because they are not equally trustworthy and the difference is the whole point
        # of fitting on your own photos.
        rows = (personal_classifier.merge_with_text(personal, ranked,
                                                    top_k=max(len(ranked), 1))
                if personal_classifier is not None else
                [{"name": r["name"], "score": r["score"], "source": "text",
                  "confidence": "text"} for r in ranked])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(r["name"]))
            it = QTableWidgetItem(f"{r['score']*100:.2f}%")
            if i == 0:
                it.setForeground(QColor("#047857"))
            self.table.setItem(i, 2, it)
            src = "your labels" if r["source"] == "your labels" else "text"
            if r.get("confidence") == "uncertain":
                src += " (uncertain)"
            elif r.get("confidence") == "weak":
                src += " (weak)"
            cell = QTableWidgetItem(src)
            if src.startswith("your labels"):
                cell.setForeground(QColor("#1d4ed8" if r.get("confidence") == "confident"
                                          else "#b45309"))
            else:
                cell.setForeground(QColor("#6b7280"))
            self.table.setItem(i, 3, cell)
        # Remember the ACTUAL ranking, tied to the photo it was computed for, so the
        # vision model can be given the classifier's opinion. Previously it was handed
        # the first 40 names of the INPUT list and told those were the classifier's
        # proposals — which is both false and can omit the true species entirely.
        self.last_ranking = [(r["name"], float(r["score"])) for r in rows]
        self.last_ranking_path = self.image_path
        top = rows[0]["name"] if rows else "?"
        self.log.appendPlainText(
            f"BioCLIP 2: {len(rows)} shown of {out.get('candidates_scored')} scored, "
            f"{out.get('seconds',0):.1f}s — top: {top}")
        if out.get("personal_error"):
            self.log.appendPlainText(
                f"Personal classifier failed on this photo, text ranking unaffected: "
                f"{out['personal_error']}")
        elif personal:
            margin = personal.get("margin", 0.0)
            verdict = {"confident": "confident", "uncertain": "UNCERTAIN",
                       "weak": "WEAK"}.get(personal.get("confidence"), "?")
            self.log.appendPlainText(
                f"Your labels: {personal.get('top')} — {verdict}, margin {margin:.3f}\n"
                f"  a margin under {personal_classifier.CONFIDENT_MARGIN} means the "
                f"closest label is not clearly better than another, so check it; the "
                f"ranked rows below it come from the candidate-list text ranking and "
                f"cover species you have not labelled.")
        self.statusBar().showMessage(f"Top pick: {top}", 20000)

    def _ranking_for_vision(self) -> list[tuple[str, float]]:
        """The BioCLIP ranking, but only if it was computed for the current photo."""
        if self.last_ranking and self.last_ranking_path == self.image_path:
            return self.last_ranking
        return []

    def _context_for_vision(self) -> str:
        """Where and what-kind context, for the vision model.

        Deliberately NOT the candidate list: location and group narrow the space
        without naming an answer, so the model stays independent on the species while
        knowing it is looking at, say, Philippine reef life rather than Swedish garden
        life. Framed as user-supplied and fallible — the search area itself can be
        wrong (a place name can resolve to a same-named town a thousand km away), so
        the model is told to contradict it if the photo disagrees.
        """
        bits: list[str] = []
        try:
            lat = float(self.lat.text().strip())
            lon = float(self.lon.text().strip())
            where = f"{lat:.3f}, {lon:.3f}"
            if self.place_label:
                where += f" ({self.place_label})"
            where += f", within roughly {self.radius.value()} km"
            bits.append(f"location: {where}")
        except (TypeError, ValueError):
            pass
        # No depth filter exists anywhere: GBIF's depth field is populated
        # inconsistently — 143 of 541 nudibranch species at Anilao carry no depth at
        # all — so filtering on it silently removes real species. Recorded in the
        # export so a result can be reproduced.
        bits.append("depth: not filtered (GBIF depth metadata is inconsistent)")
        # Still tell the model that habitat is unconstrained: it changes how it should
        # read the photo, even though no filter is applied.
        bits.append("habitat: not restricted — could be land or underwater")
        try:
            label = self.group.currentText()
            taxon = None
            if gbif_candidates is not None:
                gname = gbif_candidates.group_from_spec(self.group.currentData())
                if gname:
                    taxon = gbif_candidates.GROUP_TAXA.get(gname, (None, None))[0]
            if label and not label.lower().startswith("no group"):
                bits.append(f"assumed group (the user's own guess): {label}"
                            + (f" — i.e. {taxon}" if taxon else ""))
        except Exception:
            pass
        if not bits:
            return ""
        return ("\n\nContext supplied by the user. It is a hint, not evidence, and it "
                "may be wrong — if the photo clearly contradicts it, say so and go "
                "with what you can see:\n" + "\n".join(f"  - {b}" for b in bits))

    def ask_vision(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, APP_NAME, "Open a photo first.")
            return
        if not self.vision.healthy():
            ok, msg = self.vision.start()
            if not ok:
                QMessageBox.warning(self, APP_NAME,
                                    f"Vision server unavailable:\n{msg}\n\n"
                                    "Use Vision server → Start, or check that "
                                    "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf is present.")
                return
        if self.worker:
            return
        # Reason BEFORE naming. Asking for a name first invites anchoring on whatever
        # candidates the model was shown (observed: a wrong rank-1 pick simply echoed).
        prompt = ("Look at this photo carefully. FIRST describe the diagnostic features "
                  "you can actually see — shape, colour, pattern, texture, proportions, "
                  "and anything distinctive. ONLY THEN give your best species "
                  "identification, your confidence, and what would raise it.\n\n"
                  "If the photo does not show enough detail to name a species, say so "
                  "and name the smallest group you are confident about rather than "
                  "guessing.")
        # Location / habitat / group: priors that narrow the space without naming an
        # answer, so they are always included (unlike the candidate list, which is
        # opt-in because it leaks BioCLIP's pick).
        prompt += self._context_for_vision()
        self.ask_btn.setEnabled(False)
        self.prog.setFormat("vision model thinking…")
        self.reason.setPlainText("…thinking")
        # Only send a ranking that was computed for THIS image. Switching photo after
        # Identify must not hand the model another animal's top picks.
        show_list = (self.show_list_chk.isChecked()
                     if hasattr(self, "show_list_chk") else False)
        ranking = self._ranking_for_vision() if show_list else []
        shortlist = self.last_candidates if (show_list and not ranking) else []
        self.worker = VisionWorker(int(self.cfg.get("vision_port") or VISION_PORT),
                                   self.image_path, prompt, shortlist, ranking)
        self.worker.done.connect(self._on_vision)
        self.worker.failed.connect(self._on_fail)
        self._track(self.worker)
        self.worker.start()

    def _on_vision(self, text: str, el: float) -> None:
        self.ask_btn.setEnabled(True)
        self.prog.setFormat(f"done — {el:.0f}s")
        self.reason.setPlainText(text)
        self.log.appendPlainText(f"vision model answered in {el:.0f}s ({len(text)} chars)")
        self._cross_check_vision(text)

    def _cross_check_vision(self, text: str) -> None:
        """Report where the vision model's answer sat in BioCLIP's ranking.

        The whole value of asking twice is the comparison: "vision says X, and
        BioCLIP had X at #10" tells you far more than either answer alone — and if
        the model merely restates BioCLIP's #1, that is worth knowing too, because
        it means the two answers are not independent evidence.
        """
        ranked = self.last_ranking
        if not ranked or not text:
            return
        low = text.lower()
        hits = [(i, n, s) for i, (n, s) in enumerate(ranked) if n.lower() in low]
        if not hits:
            self.log.appendPlainText(
                f"cross-check: the vision model named no species from BioCLIP's top "
                f"{len(ranked)} — a genuine disagreement, worth reading its reasoning")
            return
        i, n, s = hits[0]
        if i == 0:
            self.log.appendPlainText(
                f"cross-check: the vision model agreed with BioCLIP's top pick "
                f"({n}, {s*100:.1f}%) — not independent evidence, and if that pick is "
                "wrong both are wrong for the same reason")
        else:
            self.log.appendPlainText(
                f"cross-check: the vision model named {n}, which BioCLIP ranked "
                f"#{i + 1} ({s*100:.1f}%) — the two disagree on the top pick")
        if len(hits) > 1:
            others = ", ".join(f"{n2} #{i2 + 1}" for i2, n2, _ in hits[1:4])
            self.log.appendPlainText(f"            it also mentioned: {others}")

    def _on_fail(self, msg: str) -> None:
        self.ident_btn.setEnabled(True)
        self.ask_btn.setEnabled(True)
        self.prog.setFormat("failed")
        self.log.appendPlainText("ERROR: " + msg)
        QMessageBox.warning(self, APP_NAME, msg)

    # -- batch mode -------------------------------------------------------- #
    def run_batch(self) -> None:
        if self.batch:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of photos", self.cfg.get("last_batch_dir") or str(Path.home()))
        if not folder:
            return
        self.cfg["last_batch_dir"] = folder
        save_cfg(self.cfg)
        self._start_batch(folder, files=None)

    def run_batch_files(self) -> None:
        """Batch the photos you pick by hand, not a whole folder."""
        if self.batch:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select photos for batch",
            self.cfg.get("last_batch_dir") or self.cfg.get("last_dir") or str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)")
        if not files:
            return
        self.cfg["last_batch_dir"] = str(Path(files[0]).parent)
        save_cfg(self.cfg)
        self._start_batch(str(Path(files[0]).parent), files=files)

    def _start_batch(self, folder: str, files: list[str] | None) -> None:
        self.results = []
        self.btable.setRowCount(0)
        self.batch_btn.setEnabled(False)
        self.batch_cancel.setEnabled(True)
        self.batch = BatchWorker(
            folder,
            recursive=self.recursive_chk.isChecked(),
            auto_gps=self.autogps_chk.isChecked(),
            online=self.online_chk.isChecked(),
            fixed_candidates=self.candidates(),
            radius_km=int(self.radius.value()),
            top_k=int(self.cfg.get("top_k", 15)),
            month_override=self.month.currentData() or None,
            spec=self.group.currentData(),
            files=files,
        )
        self.batch.row.connect(self._on_batch_row)
        self.batch.progress.connect(self._on_batch_progress)
        self.batch.finished_ok.connect(self._on_batch_done)
        self.batch.failed.connect(self._on_fail)
        self._track(self.batch)
        self.log.appendPlainText(f"batch: {folder} "
                                 f"(recursive={self.recursive_chk.isChecked()}, "
                                 f"gps={self.autogps_chk.isChecked()}, "
                                 f"online={self.online_chk.isChecked()})")
        self.tabs.setCurrentWidget(self.batch_tab)      # show results as they land
        self.batch.start()

    def _cancel_batch(self) -> None:
        if self.batch:
            self.batch.cancel()
            self.log.appendPlainText("batch: cancel requested")

    def _on_batch_progress(self, done: int, total: int, name: str) -> None:
        self.prog.setValue(int(done * 100 / max(1, total)))
        self.prog.setFormat(f"batch {done}/{total} — {name}")

    def _on_batch_row(self, r: dict) -> None:
        self.results.append(r)
        i = self.btable.rowCount()
        self.btable.insertRow(i)
        meta = r.get("meta") or {}
        ranked = r.get("ranked") or []
        top = ranked[0] if ranked else None
        gps = "—" if meta.get("lat") is None else f"{meta['lat']:.3f}, {meta['lon']:.3f}"
        taken = (str(meta.get("datetime"))[:16].replace("T", " ")
                 if meta.get("datetime") else "—")
        cells = [
            r.get("name", "?"),
            taken,
            gps,
            (top["name"] if top else f"ERROR: {r.get('error') or '?'}"),
            (f"{top['score']*100:.2f}%" if top else ""),
            r.get("candidate_source", ""),
        ]
        for c, val in enumerate(cells):
            it = QTableWidgetItem(str(val))
            if c == 0:
                it.setToolTip(r.get("photo", ""))
            if c == 3 and top:
                it.setToolTip(" · ".join(f"{x['name']} ({x['score']*100:.1f}%)"
                                         for x in ranked[:5]))
                it.setForeground(QColor("#047857"))
            self.btable.setItem(i, c, it)

    def _on_batch_done(self, count: int, msg: str) -> None:
        # NOTE: do NOT clear self.batch here — this slot runs while the worker's
        # run() is still on the stack (finished_ok is emitted from inside it), and
        # dropping the last reference aborts the process. _reap() clears it on
        # QThread.finished, which fires after run() has actually returned.
        self.batch_btn.setEnabled(True)
        self.batch_cancel.setEnabled(False)
        self.prog.setValue(100)
        self.prog.setFormat("batch done")
        self.log.appendPlainText(f"batch finished: {msg}")
        self.statusBar().showMessage(msg, 20000)
        if self.results:
            # Auto-file the run so results survive without a manual export.
            out = Path(self.cfg.get("out_dir") or (Path.home() / "species_results"))
            try:
                out.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                self._write_csv(out / f"species_{stamp}.csv")
                self._write_json(out / f"species_{stamp}.json")
                self.log.appendPlainText(f"results saved to {out}")
            except Exception as exc:
                self.log.appendPlainText(f"could not auto-save results: {exc}")

    # -- copying a result / previewing a batch photo ------------------------ #
    def _species_at(self, row: int) -> tuple[str, str]:
        """(name, probability) for a row of the single-photo results table."""
        if row < 0:
            return "", ""
        nm = self.table.item(row, 1)
        prob = self.table.item(row, 2)
        return (nm.text() if nm else "", prob.text() if prob else "")

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows:
            return rows[0].row()
        return self.table.currentRow()

    def _copy_species(self) -> None:
        name, prob = self._species_at(self._selected_row())
        if not name:
            self.statusBar().showMessage("no species row selected", 8000)
            return
        QGuiApplication.clipboard().setText(name)
        self.statusBar().showMessage(f"copied “{name}” to the clipboard", 12000)

    def _copy_species_with_score(self) -> None:
        name, prob = self._species_at(self._selected_row())
        if not name:
            return
        QGuiApplication.clipboard().setText(f"{name}\t{prob}")
        self.statusBar().showMessage(f"copied “{name}  {prob}”", 12000)

    def _species_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        self.table.selectRow(row)
        name, prob = self._species_at(row)
        m = QMenu(self)
        a1 = m.addAction(f"Copy name  “{name}”")
        a1.triggered.connect(self._copy_species)
        a2 = m.addAction(f"Copy name + probability  ({prob})")
        a2.triggered.connect(self._copy_species_with_score)
        m.addSeparator()
        # Checking the identification elsewhere is the usual next step, so offer it
        # directly — the clipboard is there too, for anywhere these do not cover.
        a3 = m.addAction("Open in iNaturalist (browser)")
        a3.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(f"https://www.inaturalist.org/search?q={name}")))
        # NOT gbif.org: every deep link on that site is refused by Cloudflare on this
        # network — the homepage loads, /species/… and /occurrence/… return 403 in a real
        # browser. The API answers, so ask the API and show the verdict here.
        a4 = m.addAction("Look up in GBIF (in this app)")
        a4.triggered.connect(lambda: self.lookup_gbif(name))
        a4b = m.addAction("Open in WoRMS (browser)")
        a4b.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(
                "https://www.marinespecies.org/aphia.php?p=taxlist&searchpar="
                + name.replace(" ", "+"))))
        a5 = m.addAction("Open in Google Images")
        a5.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl(f"https://www.google.com/search?tbm=isch&q={name.replace(' ', '+')}")))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def _copy_text(self, text: str) -> None:
        """Put arbitrary text on the clipboard and say how much went."""
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        lines = text.count("\n") + 1
        self.statusBar().showMessage(
            f"copied {len(text):,} characters ({lines} line"
            f"{'s' if lines != 1 else ''}) to the clipboard", 12000)

    def _reason_menu(self, pos) -> None:
        """Copy / look-up affordances for the vision model's answer.

        Ctrl+C already copies a selection, but there was no way to take the whole
        response, and no way to follow a species name the model mentioned in prose out to
        a reference page. Treat the selected text — or the word under the cursor when
        nothing is selected — as the name to look up.
        """
        cur = self.reason.textCursor()
        selected = cur.selectedText().replace("\u2029", "\n").strip()
        if not selected:
            probe = QTextCursor(cur)
            probe.select(QTextCursor.SelectionType.WordUnderCursor)
            selected = probe.selectedText().strip()
        full = self.reason.toPlainText()
        self._build_reason_menu(selected, full).exec(self.reason.mapToGlobal(pos))

    def _build_reason_menu(self, selected: str, full: str) -> QMenu:
        """The vision pane's menu, separated from exec() so it can be tested."""
        m = QMenu(self)
        a_sel = m.addAction("Copy selection"
                            + (f"  ({len(selected)} chars)" if selected else ""))
        a_sel.setEnabled(bool(selected))
        a_sel.triggered.connect(lambda: self._copy_text(selected))
        a_all = m.addAction(f"Copy whole answer  ({len(full):,} chars)")
        a_all.setEnabled(bool(full.strip()))
        a_all.triggered.connect(lambda: self._copy_text(full))

        # Only offer the lookups when the text could plausibly be a name: a single
        # sentence or a whole paragraph is not something you can search a taxon by.
        if selected and 2 < len(selected) < 60 and "\n" not in selected:
            m.addSeparator()
            q = selected.replace(" ", "+")
            m.addAction(f"Open “{selected}” in iNaturalist (browser)").triggered.connect(
                lambda: QDesktopServices.openUrl(
                    QUrl(f"https://www.inaturalist.org/search?q={q}")))
            m.addAction(f"Look up “{selected}” in GBIF (in this app)").triggered.connect(
                lambda: self.lookup_gbif(selected))
            m.addAction(f"Open “{selected}” in WoRMS (browser)").triggered.connect(
                lambda: QDesktopServices.openUrl(QUrl(
                    "https://www.marinespecies.org/aphia.php?p=taxlist&searchpar=" + q)))
            m.addAction(f"Search images for “{selected}”").triggered.connect(
                lambda: QDesktopServices.openUrl(
                    QUrl(f"https://www.google.com/search?tbm=isch&q={q}")))
        return m

    def _preview_photo(self, path: str) -> None:
        """Show a photo without the side effects of load_photo().

        Deliberately does NOT apply EXIF to the location fields or touch last_dir:
        clicking through a batch must not silently rewrite the search area or the
        config. It does set image_path, so Identify acts on what you can see.
        """
        pm = QPixmap(path)
        if pm.isNull():
            return
        self.image_path = path
        self.view.setPixmap(pm.scaled(self.view.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        self.path_lbl.setText(f"{Path(path).name}   ({pm.width()}×{pm.height()})")
        self.statusBar().showMessage(f"preview: {path}", 8000)

    def _on_batch_select(self, row: int, col: int, prev_row: int, prev_col: int) -> None:
        """Preview the photo of the row just selected in the batch table."""
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]
        photo = r.get("photo")
        if not photo:
            return
        self._preview_photo(photo)
        top = (r.get("ranked") or [{}])[0].get("name")
        if top:
            self.statusBar().showMessage(
                f"{Path(photo).name} — most likely {top} "
                f"({(r.get('ranked') or [{}])[0].get('score', 0) * 100:.1f}%)", 12000)

    def _open_batch_row(self, item: QTableWidgetItem) -> None:
        r = self.results[item.row()] if 0 <= item.row() < len(self.results) else None
        if not r:
            return
        self.load_photo(r["photo"])
        ranked = r.get("ranked") or []
        self.table.setRowCount(len(ranked))
        for i, x in enumerate(ranked):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(x["name"]))
            it = QTableWidgetItem(f"{x['score']*100:.2f}%")
            if i == 0:
                it.setForeground(QColor("#047857"))
            self.table.setItem(i, 2, it)
        meta = r.get("meta") or {}
        self.log.appendPlainText(
            f"{r['name']}: {ranked[0]['name'] if ranked else '?'} — "
            f"taken {meta.get('datetime') or '—'}, "
            f"gps {meta.get('lat') or '—'}, cam {meta.get('camera') or '—'}")

    # -- export ----------------------------------------------------------- #
    @staticmethod
    def _flatten(r: dict) -> dict:
        meta = r.get("meta") or {}
        ranked = r.get("ranked") or []
        row = {
            "photo": r.get("photo", ""),
            "filename": r.get("name", ""),
            "taken": meta.get("datetime") or "",
            "latitude": meta.get("lat") if meta.get("lat") is not None else "",
            "longitude": meta.get("lon") if meta.get("lon") is not None else "",
            "camera": meta.get("camera") or "",
            "candidate_source": r.get("candidate_source", ""),
            "candidates_scored": r.get("candidates", ""),
            "seconds": round(r.get("seconds", 0) or 0, 2),
            "error": r.get("error") or "",
        }
        for n in (1, 2, 3):
            hit = ranked[n - 1] if len(ranked) >= n else None
            row[f"top{n}_species"] = hit["name"] if hit else ""
            row[f"top{n}_probability"] = f"{hit['score']*100:.2f}%" if hit else ""
        return row

    def _write_csv(self, path: Path) -> None:
        rows = [self._flatten(r) for r in self.results]
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _write_json(self, path: Path) -> None:
        payload = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engine": "BioCLIP 2 (local)",
            "photos": self.results,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def export(self, fmt: str) -> None:
        if not self.results:
            QMessageBox.information(self, APP_NAME,
                                    "No batch results yet — run a batch folder first.")
            return
        out = Path(self.cfg.get("out_dir") or (Path.home() / "species_results"))
        out.mkdir(parents=True, exist_ok=True)
        default = out / f"species_{time.strftime('%Y%m%d_%H%M%S')}.{fmt}"
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}",
                                              str(default),
                                              "CSV (*.csv)" if fmt == "csv" else "JSON (*.json)")
        if not path:
            return
        try:
            if fmt == "csv":
                self._write_csv(Path(path))
            else:
                self._write_json(Path(path))
            self.log.appendPlainText(f"exported {len(self.results)} rows to {path}")
            self.statusBar().showMessage(f"exported to {path}", 15000)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Export failed: {exc}")

    def closeEvent(self, e) -> None:
        save_cfg(self.cfg)
        super().closeEvent(e)


def selftest(cfg: dict) -> int:
    print(APP_NAME, "selftest")
    ok = True
    if speciesid is None:
        print("  speciesid.py  : MISSING"); ok = False
    else:
        st = speciesid.BioCLIP().status()
        for k, v in st.items():
            print(f"  bioclip {k:12}: {v}")
        ok &= bool(st["ready"])
    vs = VisionServer(int(cfg.get("vision_port") or VISION_PORT),
                      cfg.get("vision_model"), cfg.get("vision_mmproj"))
    print(f"  vision port   : {vs.port}")
    print(f"  vision model  : {vs.model.name}")
    print(f"  vision mmproj : {vs.mmproj.name}")
    print(f"  vision files  : {'ok' if vs.files_ok()[0] else 'MISSING ' + str(vs.files_ok()[1])}")
    print(f"  vision running: {vs.healthy()}"
          + (f" (serving ours: {vs.serving_this()})" if vs.healthy() else ""))
    if vision_models is not None:
        try:
            found = vision_models.discover([Path.home() / "Models"])
            usable = [o for o in found if o.vision_ok and o.projector]
            print(f"  vision options: {len(usable)} "
                  f"({', '.join(o.model.name[:34] for o in usable)})")
        except Exception as exc:
            print(f"  vision options: scan failed ({exc})")
    print(f"  online toggle : {cfg.get('online')}")
    print("  selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--vision-port", type=int)
    ap.add_argument("--image")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.vision_port:
        cfg["vision_port"] = args.vision_port
    if args.selftest:
        return selftest(cfg)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow(cfg)
    win.show()
    if args.image:
        QTimer.singleShot(400, lambda: win.load_photo(args.image))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
