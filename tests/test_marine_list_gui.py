"""The in-app path: add a species to the marine list, and see it load with the group.

Skips cleanly when PyQt6 is absent, like the other GUI suites. Works on a temporary copy of the
group files, so a test never touches real data.
"""

import os
import pathlib
import shutil
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox
except Exception:                                   # pragma: no cover
    print("  SKIP  PyQt6 not available")
    raise SystemExit(0)

fails = 0


def check(label, ok, detail: object = ""):
    global fails
    if ok:
        print(f"  ok   {label}")
    else:
        fails += 1
        print(f"  FAIL {label}   {str(detail)[:300]}")


work = pathlib.Path(tempfile.mkdtemp())
# a config in the temp dir, so nothing touches ~/.config
cfg = work / "species_studio.json"
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
os.environ["SPECIES_CFG"] = str(cfg)

import species_gui as SG
import marine_additions as MA

SG.MARINE_GROUPS = [("nudibranchs only", "nudibranchs.txt")]
SG.MARINE_LISTS = work / "marine_lists"
SG.MARINE_LISTS.mkdir()
(SG.MARINE_LISTS / "nudibranchs.txt").write_text("Doto rosea\nDoto rosacea\n")

app = QApplication.instance() or QApplication([])
win = SG.MainWindow(SG.load_cfg())
win.resize(1400, 900)
win.show()
app.processEvents()

print("1. adding from the app writes the additions file, not the generated list")
generated_before = (SG.MARINE_LISTS / "nudibranchs.txt").read_text()
win.add_to_marine_list("Phyllodesmium poindimiei")
add_file = MA.additions_path(SG.MARINE_LISTS)
check("the additions file exists", add_file.is_file(), add_file)
check("it holds the species", "Phyllodesmium poindimiei" in add_file.read_text())
check("the generated group file was NOT modified",
      (SG.MARINE_LISTS / "nudibranchs.txt").read_text() == generated_before)
check("it is also in the box already, so it takes effect at once",
      "Phyllodesmium poindimiei" in win.candidates(), win.candidates())

print("2. junk is refused with a reason and nothing is written")
win.add_to_marine_list("OI3A2349")
check("refused", "Not added" in win.statusBar().currentMessage(),
      win.statusBar().currentMessage())
check("still one entry", len(MA.load(add_file)) == 1, MA.load(add_file))

print("3. loading the group unions the addition in")
picked = {}
QInputDialog.getItem = staticmethod(
    lambda *a, **k: (a[3][0] if len(a) > 3 else "nudibranchs only", True))
win.cand_edit.setPlainText("")
win._update_cand_count()
win.load_marine_group()
app.processEvents()
names = win.candidates()
check("group names are there", {"Doto rosea", "Doto rosacea"} <= set(names), names)
check("and so is the species you added", "Phyllodesmium poindimiei" in names, names)
check("the count reports all of them", len(names) == 3, len(names))
check("the log says how many are yours", "of yours" in win.log.toPlainText(),
      win.log.toPlainText()[-160:])

print("4. regenerating the lists keeps the addition")
(SG.MARINE_LISTS / "nudibranchs.txt").write_text("Doto rosea\nDoto rosacea\nDoto ostenta\n")
win.cand_edit.setPlainText("")
win._update_cand_count()
win.load_marine_group()
app.processEvents()
names = win.candidates()
check("the regenerated name appears", "Doto ostenta" in names, names)
check("the user's addition survives", "Phyllodesmium poindimiei" in names, names)

print("5. the automatic check: added only when no list already holds it")
win.log.clear()
win.ensure_in_marine_list("Doto rosea", "test")
check("a species already in the group is not written",
      "no addition needed" in win.log.toPlainText(), win.log.toPlainText()[-160:])
check("and the additions file still holds only the earlier entry",
      MA.load(add_file) == ["Phyllodesmium poindimiei"], MA.load(add_file))

win.log.clear()
win.ensure_in_marine_list("Nembrotha chamberlaini", "your labels recognised it")
check("a species in no list is added",
      MA.load(add_file) == ["Phyllodesmium poindimiei", "Nembrotha chamberlaini"],
      MA.load(add_file))
check("the log explains why it was added", "in none of your lists" in win.log.toPlainText(),
      win.log.toPlainText()[-200:])

win.ensure_in_marine_list("Nembrotha chamberlaini")
check("a second check does not duplicate", len(MA.load(add_file)) == 2, MA.load(add_file))

print("6. with automatic adding off, it only reports")
win.log.clear()
win.cfg["auto_add_to_marine_list"] = False
# A species that is in NO list at this point: "Doto ostenta" was written into the temp
# nudibranchs file by section 4, so using it here tested the wrong branch.
win.ensure_in_marine_list("Trapania toddi", "you taught it")
check("nothing was written", "Trapania toddi" not in MA.load(add_file), MA.load(add_file))
check("the log says the species is missing from every list",
      "in none of your lists" in win.log.toPlainText(), win.log.toPlainText()[-160:])
win.cfg["auto_add_to_marine_list"] = True

print("7. the candidate list stays primary; additions are opt-in")
win.cfg["include_my_additions"] = False
win.include_additions_chk.setChecked(False)
win._set_candidates(["Doto rosea", "Doto rosacea"], replace=True, source="GBIF, 100 km")
app.processEvents()
check("switch off: exactly the loaded list", win.candidates() == ["Doto rosea", "Doto rosacea"],
      win.candidates())
check("nothing marked as yours", win._additions_in_play == set(), win._additions_in_play)
check("the count names the source only", "+0 of yours" not in win.cand_count.text()
      and "GBIF" in win.cand_count.text(), win.cand_count.text())

win.include_additions_chk.setChecked(True)
app.processEvents()
names_now = win.candidates()
check("switch on: the additions join the pool",
      {"Phyllodesmium poindimiei", "Nembrotha chamberlaini"} <= set(names_now), names_now)
check("the count admits how many are yours", "+2 of yours" in win.cand_count.text(),
      win.cand_count.text())
check("the setting is remembered in the config",
      win.cfg.get("include_my_additions") is True, win.cfg.get("include_my_additions"))

print("8. a result from your additions says so")
win.image_path = "/tmp/x.jpg"
win._on_bio({"ranked": [{"name": "Phyllodesmium poindimiei", "score": 0.31},
                        {"name": "Doto rosea", "score": 0.12}],
             "scope": set(names_now), "candidates_scored": len(names_now), "seconds": 0.1})
app.processEvents()
rows = {win.table.item(r, 1).text(): win.table.item(r, 3).text()
        for r in range(win.table.rowCount())}
check("the addition is marked, not passed off as the list",
      rows.get("Phyllodesmium poindimiei") == "text · your list", rows)
check("a list species stays plain", rows.get("Doto rosea") == "text", rows)

win.include_additions_chk.setChecked(False)
app.processEvents()
check("switching off drops them again",
      "Phyllodesmium poindimiei" not in win.candidates(), win.candidates())

print("9. a label answer outside both list and additions is still marked NOT in list")
win._on_bio({"ranked": [{"name": "Doto rosea", "score": 0.4}],
             "personal": {"ranked": [("Trapania toddi", 0.95, 0.3)], "top": "Trapania toddi",
                          "score": 0.95, "margin": 0.3, "confidence": "confident"},
             "scope": {"Doto rosea", "Doto rosacea"},
             "candidates_scored": 2, "seconds": 0.1})
app.processEvents()
rows = {win.table.item(r, 1).text(): win.table.item(r, 3).text()
        for r in range(win.table.rowCount())}
check("the learned species is flagged as not from the list",
      rows.get("Trapania toddi") == "your labels · NOT in list", rows)

print("10. a marine-list load always brings its additions")
win.include_additions_chk.setChecked(False)
app.processEvents()
win.cand_edit.setPlainText("")
win.load_marine_group()
app.processEvents()
check("your marine species are part of the marine list",
      "Phyllodesmium poindimiei" in win.candidates(), win.candidates())
_mine = len(MA.load(add_file))
check("and are marked as yours", f"+{_mine} of yours" in win.cand_count.text(),
      (win.cand_count.text(), _mine))

shutil.rmtree(work, ignore_errors=True)
print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
