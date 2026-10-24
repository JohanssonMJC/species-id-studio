#!/usr/bin/env python3
"""Propose species for unlabelled photos, so a human only has to correct mistakes.

The bottleneck is examples per species, not photos: 109 labels gave 100 species one
example each, and a classifier cannot be judged on one example. This uses the labelled
photos as prototypes, classifies the whole library, and keeps only the confident,
high-margin proposals — a wrong proposal costs the human a rename, so the filter matters
more than the recall.

Selection rules, each learned from a failure mode:
  - skip the photos that are already labelled (matched by fingerprint, not by name);
  - require a MARGIN over the best different species, not just a high similarity: a photo
    of a coral can be "very similar" to a nudibranch prototype in absolute terms, and only
    the margin distinguishes "confident" from "closest of a bad lot";
  - cap per species, so 150 proposals cover many species instead of piling onto the few
    with distinctive prototypes;
  - drop near-duplicates of an existing label (cosine > 0.97): a second crop of the same
    frame teaches nothing.

Accepting is the default: leave a proposal as it is and it becomes training data. Rename
the wrong ones and delete anything that is not the animal at all.

  python bootstrap_confirm.py <labelled_folder> --library <root> --want 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from species_labels import PHOTO_EXTS, load_labels  # noqa: E402

DEFAULT_MODEL = pathlib.Path.home() / "Models/speciesid/bioclip-2"


def fingerprint(p: pathlib.Path, head: int = 262144) -> str:
    st = p.stat()
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        h.update(fh.read(head))
    return f"{st.st_size}:{h.hexdigest()}"


def library_index(root: pathlib.Path) -> dict[str, str]:
    cache = root / "_labelling" / "library_index.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    idx: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in PHOTO_EXTS and not p.relative_to(root).parts[0].startswith("_"):
            try:
                idx.setdefault(fingerprint(p), str(p))
            except Exception:
                pass
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled", type=pathlib.Path)
    ap.add_argument("--library", type=pathlib.Path,
                    default=pathlib.Path.home() / "Pictures/uw_photos_for_fit_centroid")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--want", type=int, default=150)
    ap.add_argument("--per-species", type=int, default=3)
    ap.add_argument("--min-margin", type=float, default=0.04)
    ap.add_argument("--dupe-sim", type=float, default=0.97)
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    a = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image as PILImage, ImageDraw, ImageFont

    labels, unlabelled = load_labels(a.labelled)
    species = sorted(labels)
    lab_paths = [p for sp in species for p in labels[sp]]
    lab_y = np.array([species.index(sp) for sp in species for _ in labels[sp]])
    print(f"  prototypes: {len(lab_paths)} photos, {len(species)} species")

    model, _, pre = open_clip.create_model_and_transforms(
        "local-dir:" + str(a.model), device="cpu")
    model.eval()

    def embed(paths):
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 16):
                ims = []
                for p in paths[i:i + 16]:
                    try:
                        ims.append(pre(PILImage.open(p).convert("RGB")))
                    except Exception:
                        pass
                if ims:
                    f = model.encode_image(torch.stack(ims))
                    out.append((f / f.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(out, axis=0)

    P = embed(lab_paths)                     # prototypes
    print(f"  prototype embeddings {P.shape}")

    # library embeddings, cached beside the photo library
    cache = a.library / "_labelling" / "embeddings.npz"
    if not cache.is_file():
        return sys.exit(f"no library embeddings at {cache} — run the label shortlist first")
    z = np.load(cache, allow_pickle=True)
    X = z["X"].astype(np.float32)
    lib_paths = sorted(
        p for p in a.library.rglob("*")
        if p.suffix.lower() in PHOTO_EXTS
        and not p.relative_to(a.library).parts[0].startswith("_"))
    if len(lib_paths) != len(X):
        return sys.exit(f"embedding cache has {len(X)} rows but the library has "
                        f"{len(lib_paths)} photos — re-run the shortlist to refresh it")
    print(f"  library {X.shape}")

    # never propose a photo that is already labelled
    idx = library_index(a.library)
    labelled_originals = set()
    for p in lab_paths:
        fp = fingerprint(p)
        if fp in idx:
            labelled_originals.add(idx[fp])
    skip = np.array([str(p) in labelled_originals for p in lib_paths])
    print(f"  {int(skip.sum())} library photos are already labelled; excluded")

    # classify: top-1 label, its similarity, and the margin over the best OTHER species
    sims = X @ P.T                          # (n_lib, n_proto)
    best_lab, best_sim, margin = np.zeros(len(X), int), np.zeros(len(X)), np.zeros(len(X))
    for i in range(len(X)):
        s = sims[i]
        order = np.argsort(-s)
        j = order[0]
        best_lab[i] = lab_y[j]
        best_sim[i] = s[j]
        for jj in order[1:]:
            if lab_y[jj] != lab_y[j]:
                margin[i] = s[j] - s[jj]
                break

    # drop near-duplicates of an existing label
    proto_best = P.max(axis=0)             # unused, kept for clarity
    dupe = np.array([float((P @ X[i]).max()) >= a.dupe_sim for i in range(len(X))])

    ok = (~skip) & (margin >= a.min_margin) & (~dupe)
    print(f"  passing filters: {int(ok.sum())} "
          f"(margin >= {a.min_margin}, not a near-duplicate)")

    # rank by margin, cap per species, take --want
    order = np.argsort(-margin)
    per: dict[int, int] = {}
    chosen = []
    for i in order:
        if not ok[i]:
            continue
        c = int(best_lab[i])
        have = sum(1 for j, _ in chosen if int(best_lab[j]) == c)
        if have >= a.per_species:
            continue
        chosen.append((i, margin[i]))
        per[c] = per.get(c, 0) + 1
        if len(chosen) >= a.want:
            break
    print(f"  selected {len(chosen)} proposals across {len(per)} species "
          f"(margin {min(m for _, m in chosen):.3f} .. {max(m for _, m in chosen):.3f})")

    out = a.out or (a.labelled.parent / "_confirm")
    tgt = out / "to_confirm"
    if tgt.exists():
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True)
    manifest = []
    for n, (i, m) in enumerate(chosen, start=1):
        sp = species[int(best_lab[i])]
        safe = sp.replace(" ", "_")
        dest = tgt / f"{safe}__prop{n:03d}{lib_paths[i].suffix.lower()}"
        shutil.copy2(lib_paths[i], dest)
        manifest.append({"file": dest.name, "species": sp, "margin": round(float(m), 4),
                         "sim": round(float(best_sim[i]), 4), "source": str(lib_paths[i])})
    (out / "proposals.json").write_text(json.dumps(manifest, indent=1))

    # contact sheets: proposed name printed on each cell, so review needs no file opening
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    cell, cols, per_sheet = 330, 5, 20
    sheets = []
    for s in range(0, len(manifest), per_sheet):
        batch = manifest[s:s + per_sheet]
        rows = (len(batch) + cols - 1) // cols
        canvas = PILImage.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
        d = ImageDraw.Draw(canvas)
        for k, m in enumerate(batch):
            try:
                im = PILImage.open(tgt / m["file"]).convert("RGB")
            except Exception:
                continue
            im.thumbnail((cell - 8, cell - 60))
            cx, cy = (k % cols) * cell, (k // cols) * cell
            canvas.paste(im, (cx + (cell - im.width) // 2, cy + 44))
            d.text((cx + 6, cy + 4), m["species"][:26], fill=(255, 235, 120), font=font)
            d.text((cx + 6, cy + 24), f"margin {m['margin']:.3f}",
                   fill=(150, 200, 150), font=font)
        p = out / f"sheet_{s // per_sheet + 1}.jpg"
        canvas.save(p, quality=88)
        sheets.append(p)

    print(f"\n  proposals -> {tgt}")
    print(f"  contact sheets: {', '.join(p.name for p in sheets)}")
    print(f"  manifest: {out/'proposals.json'}")
    spread = sorted(per.values(), reverse=True)
    print(f"  per-species proposals: max {spread[0]}, median {spread[len(spread)//2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
