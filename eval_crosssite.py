#!/usr/bin/env python3
"""Cross-site evaluation using ONLY human labels.

Why this exists rather than a simple leave-one-out over everything: if the 150 bootstrap
proposals were accepted without edits, they are MACHINE labels produced by the very
prototypes used for training. Scoring them leave-one-out measures self-consistency, not
accuracy — each proposal's nearest prototype is the photo that proposed it. A number that
looks like 95% there means nothing.

So this evaluates with human ground truth on both sides:

  train   human labels from one site
  test    human labels from the OTHER site, restricted to species present in both
  vs      zero-shot text ranking on the same photos and the same candidate species

That answers the question that matters — does one photograph per species, from a
different reef, identify the animal? — without any machine label in the loop.

  python eval_crosssite.py <labelled_folder> --library <root>
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
    idx_file = a.library / "_labelling" / "library_index.json"
    if not idx_file.is_file():
        return sys.exit(f"missing {idx_file}; run fit_eval.py once to build it")
    idx = json.loads(idx_file.read_text())

    rows = []
    for sp, paths in labels.items():
        for p in paths:
            orig = idx.get(fingerprint(p))
            if not orig:
                continue
            rel = pathlib.Path(orig).relative_to(a.library)
            rows.append((sp, p, rel.parts[0] if len(rel.parts) > 1 else "(root)"))
    print(f"  {len(rows)} human-labelled photos with a recovered site")
    sites = sorted({r[2] for r in rows})
    print(f"  sites: " + ", ".join(f"{s} ({sum(1 for r in rows if r[2]==s)})" for s in sites))

    model, _, pre = open_clip.create_model_and_transforms(
        "local-dir:" + str(a.model), device="cpu")
    tok = open_clip.get_tokenizer("local-dir:" + str(a.model))
    model.eval()

    def embed(paths):
        out = []
        with torch.no_grad():
            for p in paths:
                im = pre(PILImage.open(p).convert("RGB")).unsqueeze(0)
                f = model.encode_image(im)
                out.append((f / f.norm(dim=-1, keepdim=True)).squeeze(0).numpy())
        return np.stack(out)

    allp = [r[1] for r in rows]
    X = embed(allp)

    def stats(ranks):
        r = np.asarray(ranks, float)
        if not len(r):
            return "n/a (no test photos)"
        return (f"top-1 {np.mean(r == 1)*100:5.1f}%  top-5 {np.mean(r <= 5)*100:5.1f}%"
                f"  MRR {np.mean(1/r):.3f}")

    for train_site, test_site in [(sites[0], sites[1]), (sites[1], sites[0])]:
        train = [(i, sp) for i, (sp, _, si) in enumerate(rows) if si == train_site]
        test = [(i, sp) for i, (sp, _, si) in enumerate(rows) if si == test_site]
        shared = sorted({sp for _, sp in train} & {sp for _, sp in test})
        test = [(i, sp) for i, sp in test if sp in shared]
        if not shared or not test:
            print(f"\n  {train_site} -> {test_site}: no species in common, skipped")
            continue
        cls = shared
        C = np.stack([X[[i for i, sp in train if sp == c]].mean(axis=0) for c in cls])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        with torch.no_grad():
            tf = model.encode_text(tok([f"a photo of {c}" for c in cls]))
            T = (tf / tf.norm(dim=-1, keepdim=True)).numpy()

        cent, zs = [], []
        for i, sp in test:
            truth = cls.index(sp)
            cent.append(int(np.where(np.argsort(-(C @ X[i])) == truth)[0][0]) + 1)
            zs.append(int(np.where(np.argsort(-(T @ X[i])) == truth)[0][0]) + 1)
        print(f"\n  train {train_site} ({len(train)} photos) -> test {test_site} "
              f"({len(test)} photos, {len(shared)} species in common)")
        print(f"    centroid from your photos : {stats(cent)}")
        print(f"    zero-shot text match       : {stats(zs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
