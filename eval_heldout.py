#!/usr/bin/env python3
"""The honest accuracy number: prototypes from your labels, test on held-out human labels.

Requires the set produced by make_heldout.py to have been checked by a human — those
photos are never used as prototypes, so this is the one measurement in this repo that is
not circular and not underpowered.

Reports three arms on the same candidate species and the same test photos:
  centroid      average embedding per species from the training labels
  1-NN          nearest single training photo per species
  zero-shot     BioCLIP text matching, i.e. what the app does without any labels

  python eval_heldout.py <labelled_folder> --library <root>
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from species_labels import load_labels  # noqa: E402

DEFAULT_MODEL = pathlib.Path.home() / "Models/speciesid/bioclip-2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled", type=pathlib.Path,
                    help="the folder of human labels used as prototypes")
    ap.add_argument("--heldout", type=pathlib.Path, default=None,
                    help="default: <labelled>/../_heldout/to_label")
    ap.add_argument("--include-proposals", type=pathlib.Path, default=None,
                    help="optional folder of accepted bootstrap proposals, used as "
                         "extra training examples (never as test data)")
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    a = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image as PILImage

    train, train_bad = load_labels(a.labelled)
    held_dir = a.heldout or (a.labelled.parent / "_heldout" / "to_label")
    if not held_dir.is_dir():
        return sys.exit(f"no held-out folder at {held_dir} — run make_heldout.py first")
    test, test_bad = load_labels(held_dir)
    if not test:
        return sys.exit(f"nothing labelled in {held_dir}")

    extra_n = 0
    if a.include_proposals and a.include_proposals.is_dir():
        extra, _ = load_labels(a.include_proposals)
        for sp, paths in extra.items():
            if sp in train or sp in test:
                train.setdefault(sp, []).extend(paths)
                extra_n += len(paths)
    print(f"  prototypes: {sum(len(v) for v in train.values())} photos, "
          f"{len(train)} species" + (f" (+{extra_n} bootstrap photos)" if extra_n else ""))
    if train_bad or test_bad:
        print(f"  unparsed and ignored: train {len(train_bad)}, held-out {len(test_bad)}")
    n_test = sum(len(v) for v in test.values())
    print(f"  held-out: {n_test} photos, {len(test)} species")

    # candidates: every species either side knows about; a held-out species absent from
    # the training labels is still a legitimate (harder) candidate when text matching can
    # name it, and 1-NN cannot — so report it rather than silently dropping it
    cand = sorted(set(train) | set(test))
    missing = [sp for sp in test if sp not in train]
    if missing:
        print(f"  note: {len(missing)} held-out species have no training example: "
              f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")

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

    tr_paths = [p for sp in sorted(train) for p in train[sp]]
    tr_y = np.array([cand.index(sp) for sp in sorted(train) for _ in train[sp]])
    P = embed(tr_paths)
    C = np.stack([P[tr_y == c].mean(axis=0) for c in range(len(cand))
                  if c in set(tr_y)])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    C_classes = [c for c in range(len(cand)) if c in set(tr_y)]

    with torch.no_grad():
        tf = model.encode_text(tok([f"a photo of {c}" for c in cand]))
        T = (tf / tf.norm(dim=-1, keepdim=True)).numpy()

    te_paths = [p for sp in sorted(test) for p in test[sp]]
    te_y = [cand.index(sp) for sp in sorted(test) for _ in test[sp]]
    Xq = embed(te_paths)

    def stats(ranks):
        r = np.asarray(ranks, float)
        return (f"top-1 {np.mean(r == 1)*100:5.1f}%  top-5 {np.mean(r <= 5)*100:5.1f}%"
                f"  MRR {np.mean(1/r):.3f}")

    cent, nn, zs = [], [], []
    for k, i in enumerate(te_y):
        s = C @ Xq[k]
        order = [C_classes[j] for j in np.argsort(-s)]
        cent.append(order.index(i) + 1 if i in order else len(cand) + 1)

        s2 = P @ Xq[k]
        order2 = [int(tr_y[j]) for j in np.argsort(-s2)]
        nn.append(order2.index(i) + 1 if i in order2 else len(cand) + 1)

        zs.append(int(np.where(np.argsort(-(T @ Xq[k])) == i)[0][0]) + 1)

    print(f"\n=== {n_test} held-out photos, {len(cand)} candidate species ===")
    print(f"  centroid (mean of your labels) : {stats(cent)}")
    print(f"  1-NN (nearest single label)    : {stats(nn)}")
    print(f"  zero-shot (no labels at all)   : {stats(zs)}")

    # a held-out species with no training example is a different question: the label
    # methods CANNOT answer it (nothing to match against), while text ranking still can.
    # Splitting the two keeps the headline from being dragged down by an impossible case,
    # or flattered by excluding it.
    known = [k for k, sp in enumerate(sorted(test)) if sp in train]
    unknown = [k for k, sp in enumerate(sorted(test)) if sp not in train]
    if known and unknown:
        for label, idxs in (("species you have labels for", known),
                            ("species with NO training example", unknown)):
            sel = [k for k in idxs]
            print(f"\n  {label} ({len(sel)} photos):")
            print(f"    1-NN      {stats([nn[k] for k in sel])}")
            print(f"    zero-shot {stats([zs[k] for k in sel])}")

    # Photos the human DELETED instead of renaming are withdrawals, and deleting a WRONG
    # proposal removes a classifier error from the test — which inflates every number
    # above. They cannot be scored (no ground truth was recorded), so they are counted and
    # a pessimistic bound is given treating every withdrawal as an error.
    man = held_dir.parent / "manifest.json"
    if man.is_file():
        import json as _json
        issued = {m["file"] for m in _json.loads(man.read_text())}
        present = {p.name for p in held_dir.iterdir() if p.is_file()}
        withdrawn = sorted(issued - present)
        # a rename also shows up as issued-minus-present, so only count names that have
        # no obvious successor: keep it simple and report the raw number
        if withdrawn:
            print(f"\n  WITHDRAWN: {len(withdrawn)} issued photos are no longer present "
                  f"(deleted or renamed)")
            print("    Deleting a non-subject or unidentifiable photo is correct and does "
                  "not bias anything.")
            print("    Deleting a WRONG ID instead of renaming it removes an error from "
                  "the test — rename wrong proposals, delete only non-subjects.")
            print("    Upper-bound check if every withdrawal had been an error: "
                  f"1-NN would be at least "
                  f"{sum(1 for r in nn if r == 1)/(n_test+len(withdrawn))*100:.1f}%, "
                  f"centroid {sum(1 for r in cent if r == 1)/(n_test+len(withdrawn))*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
