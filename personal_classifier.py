"""A personal classifier over BioCLIP embeddings, fitted from labelled photos.

Why this rather than fine-tuning: the encoder is frozen, so fitting costs seconds, cannot
degrade the model, and needs no GPU. Measured on held-out photos of species with labels it
reaches 90% top-1 against 65% for text ranking alone; on species with no label it cannot
work at all, so text ranking stays as the fallback. The two are complementary and the app
uses both.

DESIGN RULE — never silently refuse. A rejection threshold cannot be made both safe and
useful here: measured on 22 non-subject photos and 24 verified animal photos, a margin
threshold of 0.06 would refuse 5 real animals to reject 3 non-subjects, and absolute
similarity does not separate them at all (non-subjects reached 0.932 while real animals
went as low as 0.795). Since excluding the true species costs the whole identification,
low confidence is REPORTED rather than used as a veto: the app shows the closest label,
says it is uncertain, and still runs the text ranking beside it.

The decision logic is pure numpy and testable without torch; only fit()/embedding need the
model.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

# margin = similarity of the best match minus that of the best match from a DIFFERENT
# species. Measured medians: 0.099 for real animals, 0.017 for non-subjects.
CONFIDENT_MARGIN = 0.06
UNCERTAIN_MARGIN = 0.03

MODEL_TAG = "bioclip-2"


def centroids(prototypes: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """One mean vector per species, renormalised so cosine similarity stays valid."""
    names = sorted(set(labels.tolist()))
    C = np.stack([prototypes[labels == n].mean(axis=0) for n in names])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C.astype(np.float32), names


def classify(query: np.ndarray, prototypes: np.ndarray, labels: np.ndarray,
             top_k: int = 10, use_centroids: bool = False) -> dict:
    """Rank species for one query embedding.

    Returns {"ranked": [(species, score, margin)], "top", "margin", "confidence"} where
    confidence is "confident" | "uncertain" | "weak". Never returns "unknown": refusing is
    the caller's decision, because a wrong refusal is worse than a flagged answer.
    """
    q = np.asarray(query, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    labels = np.asarray(labels)

    if use_centroids:
        C, names = centroids(np.asarray(prototypes, dtype=np.float32), labels)
        sims = C @ q
        order = np.argsort(-sims)
        ranked = [(names[i], float(sims[i]), float(sims[i] - sims[order[1]]))
                  for i in order[:top_k]] if len(order) > 1 else \
                 [(names[order[0]], float(sims[order[0]]), 0.0)]
    else:
        sims = np.asarray(prototypes, dtype=np.float32) @ q
        order = np.argsort(-sims)
        best = int(order[0])
        margin = 0.0
        for j in order[1:]:
            if labels[j] != labels[best]:
                margin = float(sims[best] - sims[j])
                break
        seen: dict[str, tuple[float, float]] = {}
        for j in order:
            lab = str(labels[j])
            if lab not in seen:
                # per-species margin: its best photo against the best photo of another
                m = margin if j == best else 0.0
                seen[lab] = (float(sims[j]), m)
        ranked = [(k, v[0], v[1]) for k, v in
                  sorted(seen.items(), key=lambda kv: -kv[1][0])[:top_k]]

    top, score, margin = ranked[0]
    confidence = ("confident" if margin >= CONFIDENT_MARGIN
                  else "uncertain" if margin >= UNCERTAIN_MARGIN else "weak")
    return {"ranked": ranked, "top": top, "score": score, "margin": margin,
            "confidence": confidence}


def merge_with_text(personal: dict | None, text_ranked: list[dict],
                    top_k: int = 10) -> list[dict]:
    """Labels first, text ranking filling the rest. Each row names its source.

    The label answer is not dropped when it is uncertain — it is shown first anyway, with
    its confidence, because a flagged candidate still beats no candidate.
    """
    out: list[dict] = []
    if personal and personal.get("ranked"):
        name, score, margin = personal["ranked"][0]
        out.append({"name": name, "score": score, "source": "your labels",
                    "confidence": personal["confidence"], "margin": margin})
    for row in text_ranked:
        name = row.get("name")
        if any(r["name"] == name for r in out):
            continue
        out.append({"name": name, "score": row.get("score", 0.0),
                    "source": "text", "confidence": "text", "margin": None})
        if len(out) >= top_k:
            break
    return out[:top_k]


def save(path: pathlib.Path, prototypes: np.ndarray, labels: np.ndarray,
         meta: dict | None = None) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prototypes=np.asarray(prototypes, dtype=np.float32),
                        labels=np.asarray(labels), meta=json.dumps(meta or {}))
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta or {}, indent=1))


def load(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)
    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}
    return z["prototypes"], z["labels"], meta


def fit(labelled_folder: pathlib.Path, engine, out_path: pathlib.Path) -> dict:
    """Embed every labelled photo with the app's own BioCLIP engine, then save prototypes.

    `engine` is a speciesid.BioCLIP instance: reusing it means no second copy of the model
    in memory, which on this machine is the difference between fitting and swapping.
    """
    import species_labels

    labels_map, unlabelled = species_labels.load_labels(labelled_folder)
    if not labels_map:
        raise ValueError(f"no labelled photos in {labelled_folder}")
    vectors, labels = [], []
    for sp in sorted(labels_map):
        for p in labels_map[sp]:
            vectors.append(np.asarray(engine.image_features(p), dtype=np.float32))
            labels.append(sp)
    P = np.stack(vectors)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    meta = {"model": MODEL_TAG, "photos": len(labels), "species": len(counts),
            "examples_per_species": counts,
            "source_folder": str(pathlib.Path(labelled_folder)),
            "unparsed_files": len(unlabelled),
            "confident_margin": CONFIDENT_MARGIN,
            "uncertain_margin": UNCERTAIN_MARGIN}
    save(out_path, P, np.asarray(labels), meta)
    return meta
