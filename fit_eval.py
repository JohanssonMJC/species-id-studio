#!/usr/bin/env python3
"""Fit a personal classifier on labelled photos, and measure it honestly.

Two things are measured, because they answer different questions:

  ZERO-SHOT   BioCLIP's own ranking of the 100 candidate species by text similarity.
              This is what the app does today, so it is the baseline the adaptation has
              to beat.

  CENTROID    A classifier fitted on the labelled embeddings. Evaluated leave-one-out:
              each photo is classified by centroids built from the OTHER photos. With
              mostly one example per species that means the held-out class usually has
              no examples left, so those cases are counted and reported separately
              rather than scored as successes or failures — the honest arithmetic, not
              a flattering one.

Also reports a per-site breakdown, matched by file fingerprint against the library, so
a classifier trained on one reef and tested on another is visible as it happens.

  python fit_eval.py <labelled_folder> --library <photo_library_root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from species_labels import PHOTO_EXTS, load_labels  # noqa: E402

DEFAULT_MODEL = pathlib.Path.home() / "Models/speciesid/bioclip-2"


def fingerprint(p: pathlib.Path, head: int = 262144) -> str:
    st = p.stat()
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        h.update(fh.read(head))
    return f"{st.st_size}:{h.hexdigest()}"


def library_index(root: pathlib.Path) -> dict[str, str]:
    """fingerprint -> path, cached INSIDE the photo folder so scratch pruning cannot
    delete it. Excludes this tool's own output folders."""
    cache = root / "_labelling" / "library_index.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    idx: dict[str, str] = {}
    paths = [p for p in root.rglob("*")
             if p.suffix.lower() in PHOTO_EXTS
             and not p.relative_to(root).parts[0].startswith("_")]
    for i, p in enumerate(paths, 1):
        try:
            idx.setdefault(fingerprint(p), str(p))
        except Exception:
            pass
        if i % 1000 == 0:
            print(f"\r  indexing library {i}/{len(paths)}", end="", flush=True)
    print()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled", type=pathlib.Path)
    ap.add_argument("--library", type=pathlib.Path,
                    default=pathlib.Path.home() / "Pictures/uw_photos_for_fit_centroid")
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    a = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image as PILImage

    labels, unlabelled = load_labels(a.labelled)
    species = sorted(labels)
    photos = [p for sp in species for p in labels[sp]]
    y = np.array([species.index(sp) for sp in species for _ in labels[sp]])
    print(f"  {len(photos)} labelled photos | {len(species)} species"
          + (f" | {len(unlabelled)} unparsed (ignored)" if unlabelled else ""))

    # site per photo, recovered by content matching against the library
    idx = library_index(a.library)
    sites = []
    for p in photos:
        orig = idx.get(fingerprint(p))
        if orig:
            rel = pathlib.Path(orig).relative_to(a.library)
            sites.append(rel.parts[0] if len(rel.parts) > 1 else "(root)")
        else:
            sites.append("unknown")
    sites = np.array(sites)
    print("  sites:", {s: int((sites == s).sum()) for s in sorted(set(sites))})

    model, _, pre = open_clip.create_model_and_transforms(
        "local-dir:" + str(a.model), device="cpu")
    tok = open_clip.get_tokenizer("local-dir:" + str(a.model))
    model.eval()

    with torch.no_grad():
        X = []
        for p in photos:
            im = pre(PILImage.open(p).convert("RGB")).unsqueeze(0)
            f = model.encode_image(im)
            X.append((f / f.norm(dim=-1, keepdim=True)).squeeze(0).numpy())
        X = np.stack(X)
        # same prompt style the app uses for a candidate list
        tf = model.encode_text(tok([f"a photo of {s}" for s in species]))
        T = (tf / tf.norm(dim=-1, keepdim=True)).numpy()

    def stats(ranks):
        r = np.asarray(ranks, dtype=float)
        if not len(r):
            return "n/a"
        return (f"top-1 {np.mean(r == 1)*100:5.1f}%  top-5 {np.mean(r <= 5)*100:5.1f}%"
                f"  MRR {np.mean(1/r):.3f}")

    # ---- arm 1: zero-shot text matching --------------------------------------------
    zs = [(np.argsort(-(T @ X[i])) == y[i]).argmax() + 1 for i in range(len(photos))]
    print(f"\n  ZERO-SHOT (text, {len(species)} candidates): {stats(zs)}")
    for site in sorted(set(sites)):
        m = sites == site
        print(f"    {site:<18} {stats([zs[i] for i in np.where(m)[0]])}")

    # ---- arm 2: centroids, leave-one-out ------------------------------------------
    cent_ranks, skipped = [], 0
    for i in range(len(photos)):
        keep = np.ones(len(photos), dtype=bool)
        keep[i] = False
        # a class with no remaining example cannot be predicted; count, do not score
        if not (y[keep] == y[i]).any():
            skipped += 1
            continue
        cls = sorted(set(y[keep]))
        C = np.stack([X[keep & (y == c)].mean(axis=0) for c in cls])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        order = np.argsort(-(C @ X[i]))
        preds = [cls[j] for j in order]
        cent_ranks.append(preds.index(y[i]) + 1)
    print(f"  CENTROID (leave-one-out, honest subset): {stats(cent_ranks)}")
    print(f"    scored {len(cent_ranks)} of {len(photos)}; "
          f"{skipped} skipped (their species had no other example)")

    # zero-shot restricted to the SAME photos, or the comparison is worthless: the
    # centroid arms can only score the 17 photos whose species has a second example
    scored_idx = []
    for i in range(len(photos)):
        keep = np.ones(len(photos), dtype=bool)
        keep[i] = False
        if (y[keep] == y[i]).any():
            scored_idx.append(i)
    print(f"\n  apples-to-apples on those {len(scored_idx)} photos:")
    print(f"    zero-shot   {stats([zs[i] for i in scored_idx])}")

    # ---- arm 3: nearest neighbour, same exclusions --------------------------------
    nn_ranks = []
    for i in range(len(photos)):
        keep = np.ones(len(photos), dtype=bool)
        keep[i] = False
        if not (y[keep] == y[i]).any():
            continue
        sims = X[keep] @ X[i]
        order = np.argsort(-sims)
        labs = y[keep][order]
        nn_ranks.append(int(np.where(labs == y[i])[0][0]) + 1)
    print(f"  NEAREST-1-NN (same exclusions): {stats(nn_ranks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
