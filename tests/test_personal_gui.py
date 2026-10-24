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
    from PyQt6.QtWidgets import QApplication
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

print("3. an uncertain margin is flagged in the row and explained in the log")
amb = pc.classify(np.array([1.0, 1.0, 0], dtype=np.float32),
                  np.stack([P[0], P[2]]), np.array(["Alpha beta", "Gamma delta"]))
win._on_bio({"ranked": text_ranked, "personal": amb, "candidates_scored": 2,
             "seconds": 0.1})
src = win.table.item(0, 3).text()
check("flagged", "(weak)" in src or "(uncertain)" in src, src)
log = win.log.toPlainText()
check("log explains the margin", "margin" in log.lower(), log[-200:])
check("log mentions the confidence word",
      ("WEAK" in log or "UNCERTAIN" in log), log[-200:])

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

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
