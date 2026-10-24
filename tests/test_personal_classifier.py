"""Personal-classifier decision logic. No torch, no model, no photos: CI must run this
anywhere. The point of these tests is the thing that goes wrong quietly — a classifier
that confidently names something it should have flagged."""
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import personal_classifier as pc

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  ok   {name}")
    else:
        fails += 1
        print(f"  FAIL {name}  {detail}")


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# two species, one photo each, plus a second photo of A
A1 = unit([1, 0, 0])
A2 = unit([0.97, 0.24, 0])
B1 = unit([0, 1, 0])
P = np.stack([A1, A2, B1])
L = np.array(["Alpha beta", "Alpha beta", "Gamma delta"])

print("1. 1-NN identifies the near-duplicate and reports a clean margin")
r = pc.classify(unit([0.99, 0.14, 0]), P, L)
check("top is Alpha beta", r["top"] == "Alpha beta", r["top"])
check("margin positive", r["margin"] > 0.5, r["margin"])
check("confidence confident", r["confidence"] == "confident", r["confidence"])
check("ranked has both species", len(r["ranked"]) == 2, len(r["ranked"]))

print("2. an ambiguous query is flagged, not presented as an answer")
# one photo per species, query exactly between them: the margin really is ~zero here.
# (With a second Alpha beta photo at an angle the diagonal query has a genuine 0.148
# margin — it is closer to A than to B — so this must use a one-per-species set to test
# ambiguity at all.)
P1 = np.stack([A1, B1])
L1 = np.array(["Alpha beta", "Gamma delta"])
r = pc.classify(unit([1, 1, 0]), P1, L1)
check("margin is ~zero", abs(r["margin"]) < 1e-5, r["margin"])
check("flagged weak", r["confidence"] == "weak", r["confidence"])
r = pc.classify(unit([1, 0.9, 0]), P1, L1)
check("0.074 margin lands in the confident band, as the real data calibrates it",
      r["confidence"] == "confident", (round(r["margin"], 3), r["confidence"]))

print("2b. the confidence bands are exactly where the module says they are")
# q along A; B placed so the margin (1 - cos) is a chosen number, giving exact bands
def band_for(margin: float) -> str:
    q = np.array([1.0, 0.0], dtype=np.float32)
    B = np.array([1.0 - margin, np.sqrt(1.0 - (1.0 - margin) ** 2)], dtype=np.float32)
    return pc.classify(q, np.stack([q, B]), np.array(["Alpha", "Beta"]))["confidence"]

check("margin 0.10 -> confident", band_for(0.10) == "confident", band_for(0.10))
check("margin 0.06 -> confident (boundary, inclusive)",
      band_for(0.0601) == "confident", band_for(0.0601))
check("margin 0.05 -> uncertain", band_for(0.05) == "uncertain", band_for(0.05))
check("margin 0.03 -> uncertain (boundary, inclusive)",
      band_for(0.0301) == "uncertain", band_for(0.0301))
check("margin 0.01 -> weak", band_for(0.01) == "weak", band_for(0.01))
check("thresholds are the documented ones",
      (pc.CONFIDENT_MARGIN, pc.UNCERTAIN_MARGIN) == (0.06, 0.03),
      (pc.CONFIDENT_MARGIN, pc.UNCERTAIN_MARGIN))

print("3. a query unlike anything still returns a label — refusing is the caller's job")
r = pc.classify(unit([0, 0, 1]), P, L)
check("no 'unknown' verdict exists", r["top"] in set(L.tolist()), r["top"])
check("confidence weak", r["confidence"] == "weak", r["confidence"])

print("4. margin compares against the best DIFFERENT species, not the runner-up photo")
# A2 is the runner-up overall but same species, so it must not shrink the margin
r = pc.classify(unit([1.0, 0.01, 0]), P, L)
check("margin is A-vs-B, not A1-vs-A2", r["margin"] > 0.9, r["margin"])

print("5. centroid mode averages a species instead of using one photo")
C, names = pc.centroids(P, L)
check("two centroids", len(names) == 2, names)
check("centroid normalised", abs(np.linalg.norm(C[0]) - 1.0) < 1e-5, np.linalg.norm(C[0]))
check("centroid lies between the two A photos",
      C[names.index("Alpha beta")][0] > 0.99 and
      abs(C[names.index("Alpha beta")][1] - 0.12) < 0.02,
      C[names.index("Alpha beta")])
r = pc.classify(unit([0.99, 0.14, 0]), P, L, use_centroids=True)
check("centroid mode also picks Alpha beta", r["top"] == "Alpha beta", r["top"])

print("6. merging labels with text ranking: labels first, no duplicates, sources named")
text = [{"name": "Alpha beta", "score": 0.40}, {"name": "Delta epsilon", "score": 0.30},
        {"name": "Gamma delta", "score": 0.20}]
person = pc.classify(unit([0.99, 0.14, 0]), P, L)
merged = pc.merge_with_text(person, text, top_k=5)
check("label answer first", merged[0]["name"] == "Alpha beta", merged[0])
check("source tagged", merged[0]["source"] == "your labels", merged[0]["source"])
check("label answer not duplicated by text arm",
      sum(1 for m in merged if m["name"] == "Alpha beta") == 1, merged)
check("text fills the rest", len(merged) == 3, len(merged))
check("text rows tagged", all(m["source"] == "text" for m in merged[1:]), merged)
check("top_k respected", len(pc.merge_with_text(person, text, top_k=2)) == 2)

print("7. with nothing fitted, text ranking alone is returned unchanged")
merged = pc.merge_with_text(None, text, top_k=5)
check("all text", all(m["source"] == "text" for m in merged), merged)
check("order preserved", [m["name"] for m in merged] == [t["name"] for t in text], merged)

print("8. save/load round-trip keeps prototypes, labels and metadata")
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "personal_classifier.npz"
    pc.save(p, P, L, {"photos": 3, "species": 2})
    got = pc.load(p)
    check("loaded", got is not None)
    P2, L2, meta = got
    check("prototypes identical", np.allclose(P, P2), np.abs(P - P2).max())
    check("labels identical", L.tolist() == L2.tolist(), L2)
    check("meta kept", meta.get("species") == 2, meta)
    check("sidecar written", pathlib.Path(str(p) + ".json").is_file())
    check("missing file returns None", pc.load(p.with_name("nope.npz")) is None)

print("9. fit() refuses an empty folder instead of writing an unusable classifier")
with tempfile.TemporaryDirectory() as td:
    class FakeEngine:
        def image_features(self, path):
            return np.array([1, 0, 0], dtype=np.float32)
    try:
        pc.fit(pathlib.Path(td), FakeEngine(), pathlib.Path(td) / "x.npz")
        check("raised on empty folder", False)
    except ValueError:
        check("raised on empty folder", True)
    check("nothing written", not (pathlib.Path(td) / "x.npz").exists())

print("10. fit() labels from filenames and records example counts")
with tempfile.TemporaryDirectory() as td:
    d = pathlib.Path(td)
    for name in ("Alpha_beta_01.jpg", "Alpha_beta_02.jpg", "Gamma_delta.jpg",
                 "OI3A2349.jpg"):
        (d / name).write_bytes(b"x")          # contents are irrelevant to the fake engine
    class FakeEngine:
        def image_features(self, path):
            return {"Alpha_beta_01.jpg": [1, 0, 0], "Alpha_beta_02.jpg": [0.9, 0.1, 0],
                    "Gamma_delta.jpg": [0, 1, 0]}[pathlib.Path(path).name]
    meta = pc.fit(d, FakeEngine(), d / "out.npz")
    check("only named files used", meta["photos"] == 3, meta["photos"])
    check("two species", meta["species"] == 2, meta["species"])
    check("camera filename counted as ignored", meta["unparsed_files"] == 1,
          meta["unparsed_files"])
    check("example count recorded", meta["examples_per_species"]["Alpha beta"] == 2,
          meta["examples_per_species"])

print("11. the label arm must not answer outside the group that was asked for")
# Report that produced this: group "reef & bony fishes", and a nudibranch came back as
# rank 1 because the label arm ignores the candidate list entirely.
fish = [{"name": "Chaetodon kleinii", "score": 0.41},
        {"name": "Amphiprion ocellaris", "score": 0.22}]
scope = {"Chaetodon kleinii", "Amphiprion ocellaris"}
outside = {"ranked": [("Nembrotha chamberlaini", 0.91, 0.19)], "top": "Nembrotha chamberlaini",
           "score": 0.91, "margin": 0.19, "confidence": "confident"}
merged = pc.merge_with_text(outside, fish, top_k=5, scope=scope)
check("out-of-scope label answer is dropped", "Nembrotha chamberlaini" not in
      [m["name"] for m in merged], merged)
check("text ranking leads instead", merged[0]["name"] == "Chaetodon kleinii", merged[0])
check("nothing is lost from the text arm", len(merged) == 2, len(merged))
check("the log explains it", "not in the selected group" in pc.scope_note(outside, scope),
      pc.scope_note(outside, scope))
check("no note when the answer is in scope",
      pc.scope_note({"ranked": [("Chaetodon kleinii", 0.9, 0.2)], "confidence": "confident"},
                    scope) == "")
check("no note without a scope", pc.scope_note(outside, None) == "")

print("12. a WEAK label answer is shown, but not at the top")
weak = {"ranked": [("Chromodoris annae", 0.55, 0.001)], "top": "Chromodoris annae",
        "score": 0.55, "margin": 0.001, "confidence": "weak"}
merged = pc.merge_with_text(weak, fish, top_k=5, scope=scope | {"Chromodoris annae"})
check("weak answer is kept", "Chromodoris annae" in [m["name"] for m in merged], merged)
check("text answer leads", merged[0]["source"] == "text", merged[0])
check("weak answer sits second", merged[1]["name"] == "Chromodoris annae", merged[1])
check("still flagged weak", merged[1]["confidence"] == "weak", merged[1])

print("13. agreement between the arms is not a demotion — the row is one, attributed")
agree_text = [{"name": "Chromodoris annae", "score": 0.40},
              {"name": "Chaetodon kleinii", "score": 0.20}]
merged = pc.merge_with_text(weak, agree_text, top_k=5, scope=scope | {"Chromodoris annae"})
check("one row, not two", [m["name"] for m in merged].count("Chromodoris annae") == 1, merged)
check("attributed to the labels", merged[0]["source"] == "your labels", merged[0])
check("top because both arms agree", merged[0]["name"] == "Chromodoris annae", merged[0])

print("14. confident and uncertain answers still lead; the switch can be turned off")
for conf, want_lead in (("confident", True), ("uncertain", True)):
    p = {"ranked": [("Chromodoris annae", 0.8, 0.05)], "top": "Chromodoris annae",
         "score": 0.8, "margin": 0.05, "confidence": conf}
    m = pc.merge_with_text(p, fish, top_k=5, scope=scope | {"Chromodoris annae"})
    check(f"{conf} leads", (m[0]["source"] == "your labels") == want_lead, m[0])
m = pc.merge_with_text(weak, fish, top_k=5, scope=scope | {"Chromodoris annae"},
                       demote_weak=False)
check("demote_weak=False restores the old order", m[0]["source"] == "your labels", m[0])
m = pc.merge_with_text(weak, [], top_k=5, scope=scope | {"Chromodoris annae"})
check("with no text results a weak answer is still shown", len(m) == 1 and
      m[0]["name"] == "Chromodoris annae", m)

print("15. learning one photo at a time, with provenance recorded")
import shutil as _sh
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "pc.npz"
    pc.save(p, P, L, {"photos": 3, "species": 2,
                      "examples_per_species": {"Alpha beta": 2, "Gamma delta": 1}})
    before = pc.provenance_summary(pc.load(p)[2])
    check("a classifier fitted the old way counts as all human", before["human"] == 3, before)

    rng = np.random.default_rng(1)
    novel = unit([0.6, 0.5, 0.62])                    # a different frame of Alpha beta
    r = pc.add_example(p, novel, "Alpha beta", photo="new.jpg",
                       origin="app:text", rank=2, confidence=None, margin=None)
    check("added", r["added"], r)
    check("photo count grew", r["photos"] == 4, r["photos"])
    check("species count unchanged", r["species"] == 2, r["species"])
    check("example count for the species grew", r["examples_for_species"] == 3,
          r.get("examples_for_species"))
    check("provenance separates it from human labels",
          r["provenance"]["app_lower_rank"] == 1 and r["provenance"]["human"] == 3,
          r["provenance"])

    r2 = pc.add_example(p, novel, "Alpha beta", photo="new.jpg", origin="app:text", rank=2)
    check("the same photo twice is refused", not r2["added"], r2)
    check("and says why", "near-duplicate" in r2["reason"], r2["reason"])

    r3 = pc.add_example(p, unit([0.1, 0.1, 0.99]), "New species", photo="n.jpg",
                        origin="typed")
    check("a brand-new species can be learned", r3["added"] and r3["species"] == 3, r3)
    check("typed counts separately", r3["provenance"]["typed"] == 1, r3["provenance"])

    P3, L3, meta3 = pc.load(p)
    check("labels and vectors stay in step", len(P3) == len(L3) == 5, (len(P3), len(L3)))
    check("every row has a provenance entry", len(meta3.get("sources") or []) == 5,
          len(meta3.get("sources") or []))
    top = pc.classify(novel, P3, L3)["top"]
    check("the taught photo is now recognised", top == "Alpha beta", top)

print("16. adding needs an existing classifier, and rejects nonsense")
with tempfile.TemporaryDirectory() as td:
    missing = pathlib.Path(td) / "nope.npz"
    for label, fn in (("no classifier", lambda: pc.add_example(missing, unit([1, 0, 0]),
                                                              "Alpha beta")),
                      ("empty species", lambda: pc.add_example(
                          pathlib.Path(td) / "x.npz", unit([1, 0, 0]), "  "))):
        try:
            fn()
            check(f"{label} raises", False)
        except ValueError:
            check(f"{label} raises", True)
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "pc.npz"
    pc.save(p, P, L, {"photos": 3})
    try:
        pc.add_example(p, np.zeros(3), "Alpha beta")
        check("empty embedding raises", False)
    except ValueError:
        check("empty embedding raises", True)

print("17. the provenance split is what an evaluation must respect")
meta = {"photos": 5, "sources": [{"origin": "human"}, {"origin": "human"},
                                {"origin": "app:text", "rank": 1},
                                {"origin": "app:your-labels", "rank": 1},
                                {"origin": "app:text", "rank": 7}]}
prov = pc.provenance_summary(meta)
check("two human labels", prov["human"] == 2, prov)
check("two top-pick labels", prov["app_top1"] == 2, prov)
check("one lower-rank label", prov["app_lower_rank"] == 1, prov)
check("the split adds up",
      prov["human"] + prov["app_top1"] + prov["app_lower_rank"] + prov["typed"] == 5, prov)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
