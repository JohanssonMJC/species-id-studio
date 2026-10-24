#!/usr/bin/env python3
"""Build a HELD-OUT human-labelled set, the only way to get an honest accuracy number.

Why the previous evaluations could not settle it:

  - leave-one-out over the bootstrap proposals is circular: those labels were produced by
    the same prototypes used for training, so each proposal's nearest neighbour is the
    very photo that proposed it;
  - cross-site failed for a better reason: only 4 of 100 species were photographed at both
    sites, so the test had 4 photos among 4 candidates.

What works is a set of photos whose labels come from a human and which are NEVER used as
prototypes. These are sampled at RANDOM, not by confidence — picking the confident ones
would flatter the classifier in exactly the way the earlier subset comparison did.

  python make_heldout.py <labelled_folder> --library <root> --want 40

Produces <labelled>/_heldout/to_label/ with the proposed species in each filename, and a
contact sheet. Correct the wrong ones, delete anything that is not the animal, then
  python eval_heldout.py <labelled_folder> --library <root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import shutil
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled", type=pathlib.Path)
    ap.add_argument("--library", type=pathlib.Path,
                    default=pathlib.Path.home() / "Pictures/uw_photos_for_fit_centroid")
    ap.add_argument("--want", type=int, default=40)
    ap.add_argument("--per-species", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    a = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image as PILImage, ImageDraw, ImageFont

    labels, _ = load_labels(a.labelled)
    species = sorted(labels)
    lab_paths = [p for sp in species for p in labels[sp]]
    lab_y = np.array([species.index(sp) for sp in species for _ in labels[sp]])

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

    P = embed(lab_paths)
    cache = a.library / "_labelling" / "embeddings.npz"
    z = np.load(cache, allow_pickle=True)
    X = z["X"].astype(np.float32)
    lib = sorted(p for p in a.library.rglob("*")
                 if p.suffix.lower() in PHOTO_EXTS
                 and not p.relative_to(a.library).parts[0].startswith("_"))
    if len(lib) != len(X):
        return sys.exit(f"cache rows {len(X)} != library {len(lib)}; refresh the shortlist")

    idx = json.loads((a.library / "_labelling" / "library_index.json").read_text())
    already = set()
    for p in lab_paths:
        fp = fingerprint(p)
        if fp in idx:
            already.add(idx[fp])

    # candidate pool: unlabelled photos whose nearest prototype is a species we know
    rng = random.Random(a.seed)
    pool = []
    for i, p in enumerate(lib):
        if str(p) in already:
            continue
        s = X[i] @ P.T
        j = int(np.argmax(s))
        pool.append((i, species[int(lab_y[j])]))
    rng.shuffle(pool)                     # RANDOM, never confidence-ordered

    per: dict[str, int] = {}
    chosen = []
    for i, sp in pool:
        if per.get(sp, 0) >= a.per_species:
            continue
        per[sp] = per.get(sp, 0) + 1
        chosen.append((i, sp))
        if len(chosen) >= a.want:
            break
    print(f"  {len(already)} library photos already labelled; "
          f"pool {len(pool)}; selected {len(chosen)} across {len(per)} species (random)")

    out = a.labelled.parent / "_heldout"
    tgt = out / "to_label"
    if tgt.exists():
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True)
    rows = []
    for n, (i, sp) in enumerate(chosen, start=1):
        dest = tgt / f"{sp.replace(' ', '_')}__heldout{n:03d}{lib[i].suffix.lower()}"
        shutil.copy2(lib[i], dest)
        rows.append((sp, dest))
    (out / "manifest.json").write_text(json.dumps(
        [{"file": d.name, "proposed": sp, "source": str(lib[i])}
         for (i, sp), (_, d) in zip(chosen, rows)], indent=1))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    cell, cols = 330, 5
    for s in range(0, len(rows), 20):
        batch = rows[s:s + 20]
        rr = (len(batch) + cols - 1) // cols
        canvas = PILImage.new("RGB", (cols * cell, rr * cell), (24, 24, 24))
        d = ImageDraw.Draw(canvas)
        for k, (sp, dest) in enumerate(batch):
            try:
                im = PILImage.open(dest).convert("RGB")
            except Exception:
                continue
            im.thumbnail((cell - 8, cell - 60))
            cx, cy = (k % cols) * cell, (k // cols) * cell
            canvas.paste(im, (cx + (cell - im.width) // 2, cy + 44))
            d.text((cx + 6, cy + 6), sp[:26], fill=(255, 235, 120), font=font)
            d.text((cx + 6, cy + 26), f"h{s // 20 + 1}-{k + 1}", fill=(150, 200, 150),
                   font=font)
        canvas.save(out / f"sheet_{s // 20 + 1}.jpg", quality=88)

    print(f"\n  held-out photos -> {tgt}")
    print(f"  {len(set(per))} species represented")
    print(f"  contact sheets: sheet_1.jpg..sheet_{(len(rows) + 19) // 20}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
