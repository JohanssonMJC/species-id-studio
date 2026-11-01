"""A group description must never widen the window.

The failure this pins: `_group_label_for_vision()` returned `GROUP_TAXA[group][0]`, which for
a union is a list of (name, rank) tuples. When "reef & bony fishes" became a 24-order union,
that repr went into the candidate-count label as 695 characters on one line and pushed the
window past the screen edge.

Skips cleanly when PyQt6 is absent, like the other GUI suites.
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
    from PyQt6.QtWidgets import QApplication, QMessageBox, QSizePolicy
except Exception as exc:                                  # pragma: no cover
    print(f"  SKIPPED — PyQt6 unavailable: {exc}")
    raise SystemExit(0)

TD = tempfile.mkdtemp(prefix="species_width_")
os.environ["SPECIES_CFG"] = str(pathlib.Path(TD) / "cfg.json")   # never the real config
import species_gui as SG                                          # noqa: E402

QMessageBox.information = staticmethod(lambda *a, **k: None)
app = QApplication.instance() or QApplication([])
win = SG.MainWindow(SG.load_cfg())
base = win.minimumSizeHint().width()

print(f"1. the union group gets a short label (startup window {base} px)")
fish_spec = {"taxa": [47178]}
for i in range(win.group.count()):
    if win.group.itemData(i) == fish_spec:
        win.group.setCurrentIndex(i)
        break
label = win._group_label_for_vision()
check("label is short", len(label) <= 40, f"{len(label)} chars: {label}")
check("names the first taxon", label.startswith("Perciformes"), label)
check("counts the rest rather than listing it", "23" in label, label)
check("no python repr leaked", "(" not in label and "[" not in label, label)

print("2. loading 1,990 candidates does not inflate the window")
win.cand_group = label
win._set_candidates([f"Genus species{i}" for i in range(1990)], replace=True, source="GBIF")
text = win.cand_count.text()
check("count shows the number", "1,990" in text, text)
check("count text stays short", len(text) <= 80, f"{len(text)} chars: {text}")
after = win.minimumSizeHint().width()
check("window minimum width unchanged", after <= base + 5, f"{base} -> {after}")

print("3. even an absurd group description cannot widen it")
win.cand_group = "X" * 700
win._update_cand_count()
check("text is capped", len(win.cand_count.text()) <= 90,
      len(win.cand_count.text()))
check("window minimum width still unchanged",
      win.minimumSizeHint().width() <= base + 5, win.minimumSizeHint().width())

print("4. the label wraps and yields its width")
check("word wrap on", win.cand_count.wordWrap())
# Ignored was the original choice and it was wrong: an Ignored label claims zero width, so
# with a stretch in the row the count rendered at 0 px — in the layout, invisible on screen.
# Preferred keeps the hint when there is room and shrinks (word wrapped) when there is not.
check("horizontal policy is Preferred, not Ignored",
      win.cand_count.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Preferred,
      win.cand_count.sizePolicy().horizontalPolicy())
check("minimum width zero", win.cand_count.minimumWidth() == 0)

print("5. a group with no GBIF filter says so, in the same short style")
# the message must exist for any future group that cannot be mapped
win.cand_group = ""
win._update_cand_count()
check("empty group adds no suffix", "—" not in win.cand_count.text(),
      win.cand_count.text())

print("6. the count is actually VISIBLE, and says what the list is")
win.resize(1400, 900)
win.show()
app.processEvents()
win.place_label = "Bitung, ID"
win.radius.setValue(100)
# select the group explicitly: the ambient config carries whatever was last used, and a test
# that depends on it tests the config rather than the label
for i in range(win.group.count()):
    if "nudibranch" in win.group.itemText(i).lower():
        win.group.setCurrentIndex(i)
        break
win._on_gbif([f"Nembrotha sp{i}" for i in range(828)],
             "828 species at Bitung, 100 km")
app.processEvents()
txt = win.cand_count.text()
check("the count is in the text", "828 candidates" in txt, txt)
check("the group is named", "Nudibranchia" in txt, txt)
check("the source is named", "GBIF" in txt, txt)
check("it rendered with a real width", win.cand_count.geometry().width() > 40,
      win.cand_count.geometry().width())
check("on a single line, not wrapped mid-phrase",
      win.cand_count.geometry().height() <= 24, win.cand_count.geometry().height())

print("7. no label anywhere is present but zero-width")
from PyQt6.QtWidgets import QLabel as _QL
invisible = [l.text()[:40] for l in win.findChildren(_QL)
             if l.text() and l.isVisible() and l.geometry().width() == 0]
check("every visible label with text has width", not invisible, invisible)

print("8. the candidate box cannot be squeezed away")
check("the text box has a minimum height", win.cand_edit.minimumHeight() >= 80,
      win.cand_edit.minimumHeight())
check("and gets a laid-out height", win.cand_edit.geometry().height() >= 80,
      win.cand_edit.geometry().height())
check("the count row sits below the box, not on top of it",
      win.cand_count.geometry().y() >= win.cand_edit.geometry().bottom() - 2,
      (win.cand_edit.geometry().getRect(), win.cand_count.geometry().getRect()))

print("9. a fill with no source recorded still reads sensibly")
win.cand_source = ""
win.cand_group = ""                     # a hand-typed list has no group provenance
win.cand_edit.setPlainText("\n".join(["Alpha beta", "Gamma delta"]))
win._update_cand_count()
check("plain count, no dangling separator",
      win.cand_count.text() == "2 candidates", win.cand_count.text())

print("10. clearing the list, and getting it back")
win.place_label = "Bitung, ID"
win.radius.setValue(100)
win._on_gbif([f"Nembrotha sp{i}" for i in range(828)],
             "828 species at Bitung, 100 km")
app.processEvents()
check("a clear button exists", win.clear_btn.text() == "Clear")
check("it is fixed width so it cannot grow the row", win.clear_btn.width() == 64,
      win.clear_btn.width())
before_min = win.minimumSizeHint().width()

win.clear_candidates()
app.processEvents()
check("the box is empty", win.cand_edit.toPlainText() == "", win.cand_edit.toPlainText()[:60])
check("the count reads zero", win.cand_count.text() == "0 candidates", win.cand_count.text())
check("the group and source are forgotten",
      win.cand_group == "" and win.cand_source == "",
      (win.cand_group, win.cand_source))
check("the cleared names are kept for undo", win._cleared["n"] == 828, win._cleared["n"])
check("clearing did not widen the window",
      win.minimumSizeHint().width() == before_min, win.minimumSizeHint().width())

win.restore_candidates()
app.processEvents()
check("restore brings every name back",
      win.cand_edit.toPlainText().count("\n") + 1 == 828,
      win.cand_edit.toPlainText().count("\n") + 1)
check("restore brings the source back", "GBIF" in win.cand_count.text(), win.cand_count.text())
check("the undo slot is consumed", win._cleared is None)

win.clear_candidates()          # empty it
win.clear_candidates()          # and press again
check("clearing an empty list says so instead of failing",
      "already empty" in win.statusBar().currentMessage(),
      win.statusBar().currentMessage())

print("11. sort and de-duplicate, ignoring case")
win.cand_edit.setPlainText("\n".join(["gamma delta", "Alpha beta", "alpha beta", "",
                                      "Alpha beta", "Zeta omega"]))
win._update_cand_count()
win.sort_candidates()
check("blank lines and case duplicates removed",
      win.cand_edit.toPlainText().split("\n") == ["Alpha beta", "gamma delta", "Zeta omega"],
      win.cand_edit.toPlainText().split("\n"))
check("count follows", win.cand_count.text().startswith("3 candidate"),
      win.cand_count.text())

print("12. clearing keeps the identification on screen")
win.image_path = "/tmp/x.jpg"
win._on_bio({"ranked": [{"name": "Doto rosea", "score": 0.4}],
             "candidates_scored": 3, "seconds": 0.1})
rows_before = win.table.rowCount()
win.clear_candidates()
app.processEvents()
check("the result table is untouched", win.table.rowCount() == rows_before,
      (rows_before, win.table.rowCount()))
check("so is the photo", win.image_path == "/tmp/x.jpg", win.image_path)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
