"""The app path for the personal classifier, exercised headlessly.

No model is loaded and no photograph is used: embeddings are synthetic, so this runs in CI
in about a second. It checks the parts that fail silently in a GUI — the fitted file is
found and reloaded, the label answer appears first and is labelled as such, a weak margin
is flagged rather than hidden, and switching the feature off restores text-only behaviour.

Skips cleanly when PyQt6 is absent instead of pretending to pass.
"""
import os
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  ok   {name}")
    else:
        fails += 1
        print(f"  FAIL {name}  {detail}")


try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QMessageBox
    import numpy as np
except Exception as exc:                                  # pragma: no cover
    print(f"  SKIPPED — PyQt6/numpy unavailable: {exc}")
    raise SystemExit(0)

import personal_classifier as pc

TD = tempfile.mkdtemp(prefix="species_gui_test_")
os.environ["SPECIES_CFG"] = str(pathlib.Path(TD) / "cfg.json")   # never the real config
import species_gui                                        # noqa: E402

app = QApplication.instance() or QApplication([])
win = species_gui.MainWindow(species_gui.load_cfg())

print("1. with nothing fitted, identification is text-only and says so")
check("no personal set", win._personal_set() is None)
text_ranked = [{"name": "Textus primus", "score": 0.5},
               {"name": "Textus secundus", "score": 0.3}]
win.image_path = "/tmp/not-a-real-photo.jpg"
win._on_bio({"ranked": text_ranked, "candidates_scored": 2, "seconds": 0.1})
check("4 columns", win.table.columnCount() == 4, win.table.columnCount())
check("rows shown", win.table.rowCount() == 2, win.table.rowCount())
check("all from text",
      all(win.table.item(i, 3).text().startswith("text") for i in range(2)),
      [win.table.item(i, 3).text() for i in range(2)])
check("ranking recorded for the vision handoff",
      [n for n, _ in win.last_ranking] == ["Textus primus", "Textus secundus"],
      win.last_ranking)

print("2. a fitted classifier is found, and its answer is shown first and attributed")
P = np.stack([[1, 0, 0], [0.98, 0.2, 0], [0, 1, 0]]).astype(np.float32)
P /= np.linalg.norm(P, axis=1, keepdims=True)
L = np.array(["Alpha beta", "Alpha beta", "Gamma delta"])
pc.save(species_gui.personal_path(), P, L,
        {"photos": 3, "species": 2, "confident_margin": pc.CONFIDENT_MARGIN})
win._personal_cache = (None, None)
set_ = win._personal_set()
check("loaded from the config directory", set_ is not None)
check("prototypes and labels", set_ is not None and set_[0].shape == (3, 3),
      None if set_ is None else set_[0].shape)

q = np.array([1.0, 0.1, 0], dtype=np.float32)
personal = pc.classify(q, P, L)
win._on_bio({"ranked": text_ranked, "personal": personal,
             "candidates_scored": 2, "seconds": 0.1})
check("label answer is row 1", win.table.item(0, 1).text() == personal["top"],
      win.table.item(0, 1).text())
check("row 1 attributed to your labels",
      win.table.item(0, 3).text().startswith("your labels"), win.table.item(0, 3).text())
check("text rows follow and are attributed to text",
      any(win.table.item(i, 3).text().startswith("text")
          for i in range(1, win.table.rowCount())),
      [win.table.item(i, 3).text() for i in range(win.table.rowCount())])
check("no duplicate of the label answer",
      sum(1 for i in range(win.table.rowCount())
          if win.table.item(i, 1).text() == personal["top"]) == 1)
check("vision handoff gets the merged ranking",
      win.last_ranking[0][0] == personal["top"], win.last_ranking[:2])

print("3. an uncertain margin is flagged, and a weak one does not lead")
amb = pc.classify(np.array([1.0, 1.0, 0], dtype=np.float32),
                  np.stack([P[0], P[2]]), np.array(["Alpha beta", "Gamma delta"]))
win._on_bio({"ranked": text_ranked, "personal": amb, "candidates_scored": 2,
             "seconds": 0.1})
# the label row may no longer be first: a "weak" answer is deliberately placed below the
# best text result (see personal_classifier.merge_with_text), so find it rather than
# assuming its position
rows = [(win.table.item(i, 3).text(), win.table.item(i, 1).text())
        for i in range(win.table.rowCount())]
label_rows = [(src, name) for src, name in rows if src.startswith("your labels")]
check("a label row exists", len(label_rows) == 1, rows)
label_src = label_rows[0][0] if label_rows else ""
check("flagged", "(weak)" in label_src or "(uncertain)" in label_src, label_src)
check("a weak answer does not lead", amb["confidence"] != "weak" or
      rows[0][0].startswith("text"), rows)
log = win.log.toPlainText()
check("log explains the margin", "margin" in log.lower(), log[-200:])
check("log mentions the confidence word",
      ("WEAK" in log or "UNCERTAIN" in log), log[-200:])
check("log says where it was placed", "below the text top" in log or "as the top row" in log,
      log[-200:])

print("4. turning it off restores text-only behaviour without deleting anything")
win._toggle_personal(False)
check("cfg saved off", win.cfg.get("personal_enabled") is False)
check("_personal_set now None", win._personal_set() is None)
win._on_bio({"ranked": text_ranked, "candidates_scored": 2, "seconds": 0.1})
check("first row is text again", win.table.item(0, 3).text().startswith("text"),
      win.table.item(0, 3).text())
check("fitted file still on disk", species_gui.personal_path().is_file())
win._toggle_personal(True)
check("back on", win._personal_set() is not None)

print("5. centroid mode is a stored preference, and 1-NN is the default")
check("default is 1-NN", species_gui.DEFAULTS["personal_use_centroids"] is False)
win._toggle_centroids(True)
check("toggled on", win.cfg.get("personal_use_centroids") is True)
r_nn = pc.classify(q, P, L, use_centroids=False)
r_c = pc.classify(q, P, L, use_centroids=True)
check("both modes answer", r_nn["top"] == r_c["top"] == "Alpha beta",
      (r_nn["top"], r_c["top"]))
win._toggle_centroids(False)

print("6. the fit worker writes a classifier from labelled filenames")
folder = pathlib.Path(TD) / "labelled"
folder.mkdir()
vectors = {"Alpha_beta_01.jpg": np.array([1, 0, 0], dtype=np.float32),
           "Alpha_beta_02.jpg": np.array([0.9, 0.1, 0], dtype=np.float32),
           "Gamma_delta.jpg": np.array([0, 1, 0], dtype=np.float32)}
for name in list(vectors) + ["OI3A2349.jpg"]:
    (folder / name).write_bytes(b"x")


class StubEngine:
    def image_features(self, path):
        return vectors[pathlib.Path(path).name]


orig_engine = species_gui.BioClipWorker._engine
orig_cls = species_gui.speciesid
class FakeSpeciesId:
    BioCLIP = staticmethod(lambda: StubEngine())
species_gui.speciesid = FakeSpeciesId()
out = pathlib.Path(TD) / "fitted.npz"
worker = species_gui.PersonalFitWorker(str(folder), str(out))
got = {}
worker.done.connect(lambda m: got.update(m))
worker.failed.connect(lambda m: got.update({"error": m}))
worker.run()
check("no error", "error" not in got, got.get("error"))
check("file written", out.is_file())
check("photos counted", got.get("photos") == 3, got.get("photos"))
check("species counted", got.get("species") == 2, got.get("species"))
check("camera filename ignored", got.get("unparsed_files") == 1,
      got.get("unparsed_files"))
species_gui.speciesid = orig_cls
species_gui.BioClipWorker._engine = orig_engine

print("7. the fitted file lives in a folder named after the config, not beside it")
# The bug this pins: personal_path() returned CFG_PATH.parent / "personal_classifier.npz"
# (~/.config/personal_classifier.npz) while the fit flow saved into a folder
# (~/.config/species_studio/personal_classifier.npz). Two different paths, so a fitted
# classifier was never found and every identification said "text".
pp = species_gui.personal_path()
cfg_path = species_gui.CFG_PATH
check("inside a directory, not loose in the config dir",
      pp.parent != cfg_path.parent, f"{pp} vs {cfg_path}")
check("directory named after the config file",
      pp.parent.name == cfg_path.stem, f"{pp.parent.name} != {cfg_path.stem}")
check("exact expected path", pp == cfg_path.parent / cfg_path.stem / "personal_classifier.npz",
      pp)
check("a fitted file at that path is found",
      species_gui.personal_path().is_file() and win._personal_set() is not None)

print("8. the batch path answers from your labels too, not just single photos")
win.image_path = "/tmp/batch-not-real.jpg"
win.results.clear()
win.btable.setRowCount(0)
q2 = P[1]
person2 = pc.classify(q2, P, L)
win._on_batch_row({"photo": "/tmp/b.jpg", "name": "b.jpg",
                   "meta": {"datetime": "2026-01-02T03:04:05"},
                   "candidate_source": "list", "candidates": 880,
                   "ranked": text_ranked, "personal": person2, "seconds": 0.2,
                   "error": None})
check("batch shows the label answer first",
      win.btable.item(0, 3).text() == person2["top"], win.btable.item(0, 3).text())
check("batch attributes it to your labels",
      win.btable.item(0, 5).text().startswith("your labels"), win.btable.item(0, 5).text())
check("batch score is a similarity, not a percentage",
      win.btable.item(0, 4).text().startswith("sim "), win.btable.item(0, 4).text())
flat = species_gui.MainWindow._flatten(win.results[-1])
check("csv keeps the label arm in its own columns",
      flat.get("your_labels_species") == person2["top"], flat.get("your_labels_species"))
check("csv does not present similarity as a probability",
      flat.get("your_labels_probability") is None
      and flat.get("top1_probability", "").endswith("%"), list(flat))

print("9. the single-photo table does not print similarity as a percentage either")
win._on_bio({"ranked": text_ranked, "personal": person2, "candidates_scored": 2,
             "seconds": 0.1})
check("personal row says sim", win.table.item(0, 2).text().startswith("sim "),
      win.table.item(0, 2).text())
check("text rows keep %", win.table.item(1, 2).text().endswith("%"),
      win.table.item(1, 2).text())
check("column is called Score",
      win.table.horizontalHeaderItem(2).text() == "Score",
      win.table.horizontalHeaderItem(2).text())

print("10. teaching a photo a label updates the fitted classifier")
# start from a clean fitted file in the temp config dir
pc.save(species_gui.personal_path(), P, L,
        {"photos": 3, "species": 2,
         "examples_per_species": {"Alpha beta": 2, "Gamma delta": 1}})
win._personal_cache = (None, None)
check("fitted file present", species_gui.personal_path().is_file())

vec = np.array([0.55, 0.5, 0.67], dtype=np.float32)
win.image_path = "/tmp/teach-me.jpg"
win.last_features = vec
win.last_features_path = win.image_path
source = {"origin": "app:text", "rank": 2, "confidence": None, "margin": None}
w = species_gui.TeachLabelWorker(win.image_path, vec, "Alpha beta", source)
got = {}
w.done.connect(lambda o: got.update(o))
w.failed.connect(lambda m: got.update({"error": m}))
w.run()
check("no error", "error" not in got, got.get("error"))
check("a row was added", got.get("added") and got.get("photos") == 4, got)
check("provenance records the app as the source",
      got.get("provenance", {}).get("app_lower_rank") == 1, got.get("provenance"))

win._on_teach(got)
log = win.log.toPlainText()
check("the log reports the new counts", "now 4 photos" in log, log[-300:])
check("the log shows the provenance split", "from the app's top pick" in log, log[-300:])

# the same photo again is refused, not duplicated
w2 = species_gui.TeachLabelWorker(win.image_path, vec, "Alpha beta", source)
got2 = {}
w2.done.connect(lambda o: got2.update(o))
w2.run()
check("duplicate refused", not got2.get("added"), got2)
check("no new row", len(pc.load(species_gui.personal_path())[1]) == 4)

# teaching without a fitted classifier says so instead of crashing
species_gui.personal_path().unlink()
win._personal_cache = (None, None)
errors = []
QMessageBox.information = staticmethod(
    lambda *a, **k: errors.append(a[2] if len(a) > 2 else ""))
win.teach_current("Alpha beta")
check("explains that nothing is fitted yet",
      any("no fitted classifier" in str(e).lower() for e in errors), errors)

print("11. saving a copy named after the species")
import shutil as _shutil
work = pathlib.Path(tempfile.mkdtemp())
photo = work / "OI3A2349.jpg"
photo.write_bytes(b"\xff\xd8\xff\xe0" + b"fake jpeg" * 200)
win.image_path = str(photo)
win.cfg["save_copies_min_conf"] = "not_weak"
win._on_bio({"ranked": [{"name": "Chromodoris annae", "score": 0.72}],
             "candidates_scored": 880, "seconds": 1.0})
check("top pick remembered for naming", win._top_pick()["species"] == "Chromodoris annae",
      win._top_pick())
win.save_copy_current()
made = sorted(p.name for p in (work / "_identified").glob("*.jpg"))
check("copy created under the species name", made == ["Chromodoris_annae.jpg"], made)
check("original untouched", photo.is_file() and photo.name == "OI3A2349.jpg")
rows = species_gui.save_copies.read_manifest(work / "_identified")
check("one manifest row with provenance", len(rows) == 1 and rows[0]["species"] == "Chromodoris annae",
      rows)
win.save_copy_current()
check("re-saving the same photo does not duplicate",
      sorted(p.name for p in (work / "_identified").glob("*.jpg")) == ["Chromodoris_annae.jpg"])

check("a weak label answer is refused by default",
      win._should_save("weak", "Chromodoris annae")[0] is False,
      win._should_save("weak", "Chromodoris annae"))
check("a confident one is allowed",
      win._should_save("confident", "Chromodoris annae")[0] is True)
check("a missing name is refused", win._should_save("text", "")[0] is False)
check("placeholder names are refused", win._should_save("text", "?")[0] is False)
win.cfg["save_copies_min_conf"] = "any"
check("the skip can be turned off", win._should_save("weak", "Chromodoris annae")[0] is True)
win.cfg["save_copies_min_conf"] = "not_weak"

print("12. batch copies use each row's own decision")
win.results = [
    {"photo": str(photo), "name": "a.jpg", "ranked": [{"name": "Thor amboinensis", "score": 0.5}],
     "top_species": "Thor amboinensis", "top_confidence": "text", "top_arm": "text",
     "error": None},
    {"photo": str(photo), "name": "b.jpg", "ranked": [], "top_species": None,
     "top_confidence": None, "top_arm": None, "error": None},
]
win.cfg["save_copies_dir"] = str(work / "sorted")
win.save_copies_for_batch()
check("one copy for the named row",
      sorted(p.name for p in (work / "sorted").glob("*.jpg")) == ["Thor_amboinensis.jpg"],
      list((work / "sorted").glob("*")))
check("the row records where its copy went", win.results[0].get("copy_path", "").endswith(
      "Thor_amboinensis.jpg"), win.results[0].get("copy_path"))
check("the summary says originals are untouched",
      "originals untouched" in win.log.toPlainText(), win.log.toPlainText()[-200:])
_shutil.rmtree(work)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
