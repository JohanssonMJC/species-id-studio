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
check("horizontal policy Ignored",
      win.cand_count.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored,
      win.cand_count.sizePolicy().horizontalPolicy())
check("minimum width zero", win.cand_count.minimumWidth() == 0)

print("5. a group with no GBIF filter says so, in the same short style")
# the message must exist for any future group that cannot be mapped
win.cand_group = ""
win._update_cand_count()
check("empty group adds no suffix", "—" not in win.cand_count.text(),
      win.cand_count.text())

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
