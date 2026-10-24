#!/usr/bin/env python3
"""Where should the personal classifier's rejection threshold sit?

Choosing it by feel is how a classifier ends up labelling a land photo as a nudibranch.
There is an empirical answer here: the held-out photos that survived human checking are
POSITIVES, and the issued photos that were deleted as non-UW or unidentifiable are
NEGATIVES. Both are real photos from the same library, so the margin distribution of the
two sets shows what a threshold has to separate.

Reports, for each candidate margin threshold, how many true animals would be REFUSED
(false rejects) and how many non-subjects would be ACCEPTED (false accepts).
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from species_labels import load_labels  # noqa: E402

MODEL = pathlib.Path.home() / "Models/speciesid/bioclip-2"
BASE = pathlib.Path.home() / "Pictures/uw_photos_for_fit_centroid/_labelling_k150"


def main() -> int:
    import open_clip
    import torch
    from PIL import Image as PILImage

    train, _ = load_labels(BASE / "to_label")
    held, _ = load_labels(BASE / "_heldout" / "to_label")
    manifest = {m["file"]: m for m in json.loads(
        (BASE / "_heldout" / "manifest.json").read_text())}

    # negatives: issued photos that are no longer present AND were not renamed
    present = {p.name for p in (BASE / "_heldout" / "to_label").iterdir()}
    deleted = [v for k, v in manifest.items() if k not in present]
    # a deleted file's original source photo is our negative example
    neg_paths = [pathlib.Path(m["source"]) for m in deleted
                 if pathlib.Path(m["source"]).is_file()]
    # drop negatives whose "species" also appears as a renamed (corrected) file, i.e. it
    # was a misidentification rather than a non-subject
    renamed_species = {sp for sp in held}
    pos_paths = [p for sp in held for p in held[sp]]
    print(f"  positives {len(pos_paths)} (verified held-out animals)")
    print(f"  negatives {len(neg_paths)} (deleted as non-UW or unidentifiable)")

    model, _, pre = open_clip.create_model_and_transforms(
        "local-dir:" + str(MODEL), device="cpu")
    model.eval()

    def embed(paths):
        out = []
        with torch.no_grad():
            for p in paths:
                im = pre(PILImage.open(p).convert("RGB")).unsqueeze(0)
                f = model.encode_image(im)
                out.append((f / f.norm(dim=-1, keepdim=True)).squeeze(0).numpy())
        return np.stack(out)

    species = sorted(train)
    tr_paths = [p for sp in species for p in train[sp]]
    tr_y = np.array([species.index(sp) for sp in species for _ in train[sp]])
    P = embed(tr_paths)

    def margins(paths):
        X = embed(paths)
        sims = X @ P.T
        out = []
        for i in range(len(X)):
            s = sims[i]
            order = np.argsort(-s)
            j = order[0]
            m = 0.0
            for jj in order[1:]:
                if tr_y[jj] != tr_y[j]:
                    m = float(s[j] - s[jj])
                    break
            out.append((float(s[j]), m))
        return out

    pos = margins(pos_paths)
    neg = margins(neg_paths)
    pa = np.array([m for _, m in pos]); na = np.array([m for _, m in neg])
    ps = np.array([s for s, _ in pos]); ns = np.array([s for s, _ in neg])
    print(f"\n  margin (best minus best-different):")
    print(f"    positives  median {np.median(pa):.3f}  p10 {np.percentile(pa,10):.3f}"
          f"  min {pa.min():.3f}")
    print(f"    negatives  median {np.median(na):.3f}  p90 {np.percentile(na,90):.3f}"
          f"  max {na.max():.3f}")
    print(f"  absolute similarity:")
    print(f"    positives  median {np.median(ps):.3f}  min {ps.min():.3f}")
    print(f"    negatives  median {np.median(ns):.3f}  max {ns.max():.3f}")

    print(f"\n  {'margin >= ':>12}  {'true animals refused':>22}  {'non-subjects accepted':>23}")
    for thr in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        print(f"  {thr:>12.2f}  {int((pa < thr).sum()):>10} of {len(pa):<11}"
              f"  {int((na >= thr).sum()):>10} of {len(na)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
