"""Both group selectors must offer sea snails, and they must agree on what that means.

The app has two, and they answer different questions:

  group combobox      what kind of organism is it → drives iNaturalist/GBIF *place* queries
                      (taxon filters only; the taxa are iNaturalist ids)
  Marine list…        which curated offline list → drives scoring against marine_lists/*.txt

"Sea snails, cones & cowries" was added only to the second, and the Gastropoda entry in the
first sat at the bottom labelled "snails & slugs (land and sea)" where a diver holding a
Triton shell would never look. Both now name it, and this test keeps them named.

Skips cleanly without PyQt6.
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
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QMessageBox
except Exception as exc:                                  # pragma: no cover
    print(f"  SKIPPED — PyQt6 unavailable: {exc}")
    raise SystemExit(0)

TD = tempfile.mkdtemp(prefix="species_groups_")
os.environ["SPECIES_CFG"] = str(pathlib.Path(TD) / "cfg.json")
import species_gui as SG                                          # noqa: E402
import gbif_candidates as G                                       # noqa: E402

QMessageBox.information = staticmethod(lambda *a, **k: None)
app = QApplication.instance() or QApplication([])
win = SG.MainWindow(SG.load_cfg())

labels = [win.group.itemText(i) for i in range(win.group.count())]

print("1. the place-query combobox offers it, in the sea section")
TARGET = "sea · sea snails, cones & cowries"
check("present", TARGET in labels, [l for l in labels if "snail" in l.lower()])
check("old bottom label is gone", "snails & slugs (land and sea)" not in labels,
      [l for l in labels if "snail" in l.lower()])
if TARGET in labels:
    i = labels.index(TARGET)
    before = [l for l in labels[:i] if l.startswith("sea ·")]
    after = [l for l in labels[i + 1:] if l.startswith("sea ·")]
    check("sits after clams", any("clams" in b for b in before[-3:]), before[-3:])
    check("sits before sponges", bool(after) and "sponges" in after[0], after[:2])
    tip = win.group.itemData(i, Qt.ItemDataRole.ToolTipRole)
    check("has a tooltip", bool(tip), tip)
    check("tooltip admits land snails can appear",
          "land snail" in str(tip), str(tip)[:120])
    check("tooltip points at the curated list", "Marine list" in str(tip), str(tip)[:120])

print("2. it means Gastropoda on both backends")
for i in range(win.group.count()):
    if win.group.itemText(i) == TARGET:
        win.group.setCurrentIndex(i)
        break
check("iNat taxon id is Gastropoda (47114)", win.group.currentData() == {"taxa": [47114]},
      win.group.currentData())
check("GBIF group is snails_slugs",
      G.group_from_spec(win.group.currentData()) == "snails_slugs",
      G.group_from_spec(win.group.currentData()))
check("GBIF key is Gastropoda at CLASS",
      G.GROUP_TAXA.get("snails_slugs") == ("Gastropoda", "CLASS"),
      G.GROUP_TAXA.get("snails_slugs"))
check("the label shown next to the count is short",
      len(win._group_label_for_vision()) <= 40, win._group_label_for_vision())

print("3. the curated offline list offers the same thing, under its own name")
marine = dict(SG.MARINE_GROUPS)
check("marine list has it", "sea snails, cones & cowries" in marine, list(marine)[:3])
check("it points at gastropods.txt",
      marine.get("sea snails, cones & cowries") == "gastropods.txt",
      marine.get("sea snails, cones & cowries"))
check("bivalves are separate there", marine.get("bivalves & chitons") == "bivalves.txt",
      marine.get("bivalves & chitons"))
check("nudibranchs are separate too",
      marine.get("nudibranchs & sea slugs") == "sea_slugs.txt",
      marine.get("nudibranchs & sea slugs"))

print("4. no group is listed twice in the combobox")
sea = [l for l in labels if l.startswith("sea ·")]
check("no duplicate labels", len(sea) == len(set(sea)), [l for l in sea if sea.count(l) > 1])
gastropod_entries = [l for l in labels if l == TARGET or "snail" in l.lower()]
check("exactly one gastropod entry", len(gastropod_entries) == 1, gastropod_entries)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
