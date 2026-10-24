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
import time

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
                    top_k: int = 10, scope: set[str] | None = None,
                    demote_weak: bool = True) -> list[dict]:
    """Labels first, text ranking filling the rest. Each row names its source.

    SCOPE. When `scope` is given — the candidate list actually scored — a label answer for
    a species outside it is DROPPED from the ranking (it stays in the log separately). The
    user selected "reef & bony fishes" and got a nudibranch at rank 1: the classifier
    answered a different question from the one asked, and leading with it also pushed the
    correct fish to rank 2.

    WEAK ANSWERS DO NOT LEAD. "weak" means the closest label is not clearly better than
    another label (margin < UNCERTAIN_MARGIN). An uncertain answer still leads, flagged,
    because a flagged candidate beats no candidate — but a weak one is moved below the top
    text result instead of being shown as the best available answer.
    """
    out: list[dict] = []
    pending = None          # a weak label row, waiting to sit below the best text answer
    if personal and personal.get("ranked"):
        name, score, margin = personal["ranked"][0]
        if scope is None or name in scope:
            row = {"name": name, "score": score, "source": "your labels",
                   "confidence": personal["confidence"], "margin": margin}
            if demote_weak and personal["confidence"] == "weak" and text_ranked:
                pending = row                      # keep it, but not at the top
            else:
                out.append(row)
    for row in text_ranked:
        name = row.get("name")
        if pending is not None and pending["name"] == name:
            # the text arm agrees with the label arm: one row, attributed to the labels
            out.append(pending)
            pending = None
        elif not any(r["name"] == name for r in out):
            out.append({"name": name, "score": row.get("score", 0.0),
                        "source": "text", "confidence": "text", "margin": None})
            if pending is not None:
                out.append(pending)
                pending = None
        if len(out) >= top_k:
            break
    if pending is not None and len(out) < top_k:
        out.append(pending)
    return out[:top_k]


def scope_note(personal: dict | None, scope: set[str] | None) -> str:
    """Explain, for the log, what the label arm said and why it was not used.

    Silence here would be worse than the bug: the user chose a group, so being told that
    the fitted classifier recognised something outside it is information, not noise.
    """
    if not personal or not personal.get("ranked") or scope is None:
        return ""
    name = personal["ranked"][0][0]
    if name in scope:
        return ""
    return (f"your labels recognised “{name}” ({personal.get('confidence')}, margin "
            f"{personal.get('margin', 0):.3f}) but that species is not in the selected "
            f"group, so it is not ranked here — the text ranking stands alone")


def provenance_summary(meta: dict) -> dict:
    """How many labels came from a human, and how many from the app's own suggestion.

    This split is the whole safety story for learning as you go. A label recorded because
    the app suggested the species is NOT independent evidence: the classifier is then being
    taught what it already believes, its margins rise without its accuracy doing so, and any
    evaluation on those labels measures agreement with the proposer rather than correctness.
    Counting them is what keeps that visible; it is also why eval tools score only the
    "human" ones.
    """
    sources = meta.get("sources") or []
    out = {"human": 0, "app_top1": 0, "app_lower_rank": 0, "typed": 0, "unknown": 0}
    for s in sources:
        origin = str(s.get("origin") or "human")
        key = (origin if origin in ("human", "typed") else
               "app_top1" if origin.startswith("app:") and s.get("rank") in (1, None)
               else "app_lower_rank" if origin.startswith("app:") else "unknown")
        out[key] = out.get(key, 0) + 1
    if not sources:
        out["human"] = int(meta.get("photos") or 0)
    return out


def add_example(path, embedding, species: str, *, photo: str | None = None,
                origin: str = "human", rank: int | None = None,
                confidence: str | None = None, margin: float | None = None,
                dup_cosine: float = 0.995) -> dict:
    """Add one labelled photo to a fitted classifier, without re-encoding the folder.

    The embedding is already computed for the identification, so this costs one vector and
    one small file write — a full refit over hundreds of photos is not needed to learn one.

    `origin` records where the label came from: "human" (you typed it), "typed", or
    "app:<arm>" with the rank and confidence the app had given it. A near-duplicate of an
    existing photo of the same species is refused rather than added, because otherwise
    clicking accept twice on one photo (or on two frames of the same animal) skews that
    species' neighbourhood.

    Returns {"added": bool, "reason": ..., "photos": n, "species": n, "provenance": {...}}.
    """
    loaded = load(path)
    if loaded is None:
        raise ValueError("no fitted classifier yet — fit one from a labelled folder first")
    P, L, meta = loaded
    emb = np.asarray(embedding, dtype=np.float32).ravel()
    n = np.linalg.norm(emb)
    if n == 0:
        raise ValueError("empty embedding")
    emb = emb / n
    species = " ".join(str(species).split())
    if not species:
        raise ValueError("empty species")

    sims = P @ emb
    same = np.where(np.asarray(L) == species)[0]
    if len(same):
        best = float(np.max(sims[same]))
        if best >= dup_cosine:
            return {"added": False, "reason": f"near-duplicate of an existing "
                    f"{species} photo (cosine {best:.3f})",
                    "photos": len(L), "species": len(set(L.tolist())),
                    "provenance": provenance_summary(meta)}

    sources = list(meta.get("sources") or [])
    if not sources:                                  # a classifier fitted before this
        sources = [{"origin": "human"} for _ in range(len(L))]
    sources.append({"photo": str(photo or ""), "origin": origin, "rank": rank,
                    "confidence": confidence, "margin": margin,
                    "when": time.strftime("%Y-%m-%d %H:%M:%S")})

    P2 = np.vstack([P, emb.astype(np.float32)])
    L2 = np.append(np.asarray(L), species)
    counts: dict[str, int] = {}
    for lab in L2.tolist():
        counts[lab] = counts.get(lab, 0) + 1
    meta = dict(meta)
    meta.update({"photos": len(L2), "species": len(counts),
                 "examples_per_species": counts, "sources": sources})
    save(path, P2, L2, meta)
    return {"added": True, "reason": "added", "photos": len(L2), "species": len(counts),
            "examples_for_species": counts.get(species, 0),
            "provenance": provenance_summary(meta)}


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
