#!/usr/bin/env python3
"""
Local species-identification engine.

Tier 1 — BioCLIP 2 (zero-shot taxonomic ranking)
    A CLIP-family model trained on TreeOfLife-200M (214M images, 950k+ taxa).
    It has NO fixed label set: you supply a candidate list of scientific names
    and it scores each one. That is precisely what makes it right for
    "here are the possible species" rather than a single forced answer.

Tier 2 — Qwen3-VL-8B via llama-server --mmproj (see species_gui.py)
    Describes diagnostic features and sanity-checks Tier 1's candidates.

Loading: open_clip 3.3.0 understands a `local-dir:` schema, requiring the
directory to hold `open_clip_config.json` (with a `model_cfg` key) plus a
weight file; `open_clip_model.safetensors` is its first preferred name.

Standalone:  python3 speciesid.py --image photo.jpg [--candidates file.txt] [--top 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

MODEL_DIR = Path.home() / "Models" / "speciesid" / "bioclip-2"
MMPROJ = Path.home() / "Models" / "speciesid" / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"

# A few prompt styles averaged together; BioCLIP is robust but multi-template
# ensembling measurably helps on fine-grained taxa.
PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a close-up photo of a {}",
    "a photo of a {} in its natural habitat",
)

# NOTE: there is deliberately no built-in candidate list. A hardcoded
# temperate-European starter list used to live here and silently seeded the app's
# candidate box on launch, which meant a dive photo was being scored against ten
# European garden insects and ten European birds without the user ever choosing
# that. Candidates now come from an explicit source: the marine list, a regional
# fill, or a file.


def _require_deps():
    try:
        import torch  # noqa: F401
        import open_clip  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:            # pragma: no cover
        raise SystemExit(
            f"BioCLIP needs torch + open_clip + pillow in this interpreter.\n"
            f"Install with:\n"
            f"  uv pip install --python {sys.executable} "
            f"--index-url https://download.pytorch.org/whl/cpu torch torchvision\n"
            f"  uv pip install --python {sys.executable} open_clip_torch pillow\n"
            f"(missing: {exc.name})"
        ) from exc


# --------------------------------------------------------------------------- #
# Photo metadata (EXIF)
# --------------------------------------------------------------------------- #
def _rational(v) -> float:
    try:
        return float(v)
    except Exception:
        try:
            return v.numerator / v.denominator
        except Exception:
            return 0.0


def _dms_to_deg(dms, ref) -> float | None:
    """EXIF stores GPS as (deg, min, sec) rationals plus an N/S/E/W ref."""
    try:
        d, m, s = (_rational(x) for x in dms)
        val = d + m / 60.0 + s / 3600.0
        if str(ref).upper() in ("S", "W"):
            val = -val
        return round(val, 6)
    except Exception:
        return None


def read_exif(path: str | Path) -> dict:
    """Metadata that actually helps identification.

    Returns datetime (when the photo was taken), lat/lon (where), and camera
    info. Date gives the month for seasonal candidate filtering; GPS gives the
    location for a regional candidate list — together they are the cheapest
    large accuracy win available.
    """
    out: dict = {"datetime": None, "lat": None, "lon": None, "camera": None,
                 "has_exif": False, "error": None}
    try:
        from PIL import ExifTags, Image
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            exif = im.getexif()
            if not exif:
                return out
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            out["has_exif"] = True
            for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                if tags.get(key):
                    # EXIF format "YYYY:MM:DD HH:MM:SS"
                    raw = str(tags[key]).strip()
                    try:
                        out["datetime"] = raw.replace(":", "-", 2).replace(" ", "T")
                    except Exception:
                        out["datetime"] = raw
                    break
            make = str(tags.get("Make") or "").strip()
            model = str(tags.get("Model") or "").strip()
            # EXIF Make is often already a prefix of Model ("Canon" + "Canon EOS 40D")
            if make and model:
                cam = model if model.lower().startswith(make.lower()) else f"{make} {model}"
            else:
                cam = model or make or None
            out["camera"] = cam
            try:
                gps = exif.get_ifd(ExifTags.IFD.GPSInfo) or {}
            except Exception:
                gps = tags.get("GPSInfo") or {}
            if gps:
                lat = _dms_to_deg(gps.get(2), gps.get(1))
                lon = _dms_to_deg(gps.get(4), gps.get(3))
                out["lat"], out["lon"] = lat, lon
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def month_of(meta: dict) -> int | None:
    dt = meta.get("datetime")
    if not dt or len(str(dt)) < 7:
        return None
    try:
        return int(str(dt)[5:7])
    except Exception:
        return None


# Header words that mean "this row labels the columns, not a species".
_HEADER_HINTS = ("scientific name", "scientific_name", "species", "taxon", "name",
                 "latin name", "latin_name", "binomial", "common name",
                 "common_name", "scientificname")


def parse_candidate_file(text: str) -> list[str]:
    """Parse a candidate list: one scientific name per line.

    Accepts plain text OR CSV. A CSV row is reduced to its first field — passing
    ``Erithacus rubecula,European Robin,99`` through verbatim as a "species name"
    would silently wreck every score, because BioCLIP would encode that whole
    string as a taxon and never match the photo. Blank lines, ``#`` comments,
    a leading header row, and duplicate names are all dropped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.strip("\"'").lower() in _HEADER_HINTS:
            continue
        for sep in ("\t", ";", ","):
            if sep in line:
                first = line.split(sep, 1)[0].strip().strip("\"'")
                # only accept the split if what remains still looks like a taxon
                # name (Genus, or Genus species, or Genus sp.) rather than a note
                if first and len(first.split()) <= 3:
                    line = first
                break
        line = line.strip().strip("\"'")
        if not line or line.lower() in _HEADER_HINTS:
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


class BioCLIP:
    """Lazily-loaded, cached BioCLIP 2 ranker."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        # LRU: in batch mode each photo can carry its own GPS-derived candidate
        # list, so an unbounded dict would grow one text-embedding block per photo.
        self._text_cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._text_cache_max = 8

    # -- availability ------------------------------------------------------ #
    def status(self) -> dict:
        cfg = self.model_dir / "open_clip_config.json"
        weights = self.model_dir / "open_clip_model.safetensors"
        return {
            "dir": str(self.model_dir),
            "dir_exists": self.model_dir.is_dir(),
            "config": cfg.is_file(),
            "weights": weights.is_file(),
            "weights_gb": round(weights.stat().st_size / 1e9, 2) if weights.is_file() else 0,
            "ready": cfg.is_file() and weights.is_file(),
        }

    # -- loading ----------------------------------------------------------- #
    def load(self):
        if self._model is not None:
            return self
        st = self.status()
        if not st["ready"]:
            missing = [k for k in ("config", "weights") if not st[k]]
            raise FileNotFoundError(
                f"BioCLIP 2 not ready in {self.model_dir} (missing: {', '.join(missing)}).\n"
                "Fetch it with:\n"
                "  python3 hf_download.py get imageomics/bioclip-2 "
                "open_clip_model.safetensors <dir>/open_clip_model.safetensors\n"
                "  python3 hf_download.py get imageomics/bioclip-2 open_clip_config.json <dir>/open_clip_config.json")
        _require_deps()
        import open_clip
        import torch
        torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 4))))
        # `local-dir:` is open_clip's schema for an on-disk model directory; it
        # reads open_clip_config.json (model_cfg) and finds the weights itself.
        model, _, preprocess = open_clip.create_model_and_transforms(
            f"local-dir:{self.model_dir}", device=self.device)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(f"local-dir:{self.model_dir}")
        return self

    # -- inference --------------------------------------------------------- #
    # Encoding a whole candidate list as ONE batch allocates activations
    # proportional to list length. Fine for 30 names; a multi-thousand-entry list
    # (and a marine keyword list is ~9,000) gets the process SIGTERMed by the OOM
    # killer. Chunk it — throughput is barely affected and memory stays flat.
    TEXT_CHUNK = 512

    def _text_features(self, names: list[str]):
        import torch
        key = tuple(names)
        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            return self._text_cache[key]
        prompts = [t.format(n) for n in names for t in PROMPT_TEMPLATES]
        parts = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.TEXT_CHUNK):
                toks = self._tokenizer(prompts[i:i + self.TEXT_CHUNK])
                f = self._model.encode_text(toks)
                parts.append(f / f.norm(dim=-1, keepdim=True))
            feats = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            # average the templates per species, then renormalise
            feats = feats.view(len(names), len(PROMPT_TEMPLATES), -1).mean(dim=1)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._text_cache[key] = feats
        while len(self._text_cache) > self._text_cache_max:
            self._text_cache.popitem(last=False)
        return feats

    def image_features(self, image_path: str | Path):
        """L2-normalised image embedding, as numpy. The vector the personal classifier
        matches against; exposing it means the app never encodes the same photo twice."""
        import torch
        from PIL import Image
        self.load()
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            x = self._preprocess(img).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].float().cpu().numpy()

    def rank_from_features(self, feat, candidates: list[str], top_k: int = 10) -> list[dict]:
        """Same ranking as rank(), for an embedding already computed."""
        import torch
        self.load()
        names = [c.strip() for c in candidates if c.strip()]
        if not names:
            raise ValueError("empty candidate list")
        with torch.no_grad():
            img_feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = self._text_features(names)
            logits = (self._model.logit_scale.exp() * img_feat @ txt_feat.T)
            probs = logits.softmax(dim=-1)
        order = torch.argsort(probs, descending=True)[:top_k]
        return [{"name": names[i], "score": float(probs[i]),
                 "logit": float(logits[i])} for i in order]

    def rank(self, image_path: str | Path, candidates: list[str], top_k: int = 10) -> list[dict]:
        """Rank candidate scientific names against an image. Returns a sorted list."""
        return self.rank_from_features(self.image_features(image_path), candidates, top_k)

    def describe_image(self, image_path: str | Path, candidates: list[str],
                       top_k: int = 10) -> dict:
        feat = self.image_features(image_path)
        ranked = self.rank_from_features(feat, candidates, top_k=top_k)
        top = ranked[0] if ranked else None
        return {
            "engine": "BioCLIP 2 (local)",
            "candidates_scored": len([c for c in candidates if c.strip()]),
            "top": top,
            "ranked": ranked,
            # the embedding travels with the result so the personal classifier can match
            # against it without a second pass over the same photo
            "features": feat,
        }


def load_candidates(path: str | Path | None) -> list[str]:
    """Candidate names from a file: one scientific name per line, # comments.

    Returns [] when no file is given — there is no built-in fallback list, so the
    caller must decide what to do rather than silently scoring against a default.
    """
    if not path:
        return []
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Local species ID with BioCLIP 2")
    ap.add_argument("--image", required=True)
    ap.add_argument("--candidates", help="text file, one scientific name per line")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    engine = BioCLIP()
    st = engine.status()
    print("BioCLIP 2 status:")
    for k, v in st.items():
        print(f"  {k:12}: {v}")
    if args.status:
        return 0 if st["ready"] else 1
    if not st["ready"]:
        return 1

    cands = load_candidates(args.candidates)
    if not cands:
        print("\nNo candidate list given. Pass --candidates <file> with one "
              "scientific name per line.\n"
              "There is no built-in default list: the candidates ARE the answer "
              "space, so they have to come from somewhere you chose.\n"
              "Ready-made lists are built into marine_lists/*.txt by "
              "make_marine_lists.py from a MarineLife Keywords export you supply "
              "(they are not redistributed), e.g. marine_lists/nudibranchs.txt",
              file=sys.stderr)
        return 2
    print(f"\nscoring {len(cands)} candidates against {args.image} …")
    import time
    t0 = time.time()
    out = engine.describe_image(args.image, cands, top_k=args.top)
    print(f"done in {time.time()-t0:.1f}s on {engine.device}\n")
    print(f"{'species':34} {'prob':>7}  {'logit':>7}")
    for r in out["ranked"]:
        print(f"  {r['name']:32} {r['score']*100:6.2f}%  {r['logit']:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
