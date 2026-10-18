#!/usr/bin/env python3
"""Regenerate docs/screenshot.png from the real app.

    python docs/make_screenshot.py /path/to/a/photo.jpg

Authentic by design: a real photo, the shipped nudibranch name list, and BioCLIP's
actual output. A screenshot showing numbers the app cannot produce would be worse than
no screenshot.

Uses an isolated config (SPECIES_CFG) so it never touches your real settings.
"""
import os
import pathlib
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"          # works headless, no X needed
REPO = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("SPECIES_CFG", str(REPO / ".test-tmp" / "screenshot.json"))
sys.path.insert(0, str(REPO))

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox, QTabWidget

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

import species_gui as SG

if len(sys.argv) < 2:
    sys.exit("usage: make_screenshot.py PHOTO.jpg")
photo = pathlib.Path(sys.argv[1])
if not photo.is_file():
    sys.exit(f"no such photo: {photo}")
out = REPO / "docs" / "screenshot.png"


def wait_for(pred, timeout=600, interval=300):
    loop = QEventLoop()
    ok = {"v": False}

    def tick():
        if pred():
            ok["v"] = True
            loop.quit()

    t = QTimer()
    t.timeout.connect(tick)
    t.start(interval)
    QTimer.singleShot(int(timeout * 1000), loop.quit)
    loop.exec()
    t.stop()
    return ok["v"]


app = QApplication([])
win = SG.MainWindow(SG.load_cfg())
win.resize(1400, 900)
win.show()
for _ in range(8):
    app.processEvents()

win.load_photo(str(photo))
app.processEvents()

# The shipped nudibranch list, exactly as the marine-list path would load it.
names = [l.strip() for l in
         (REPO / "marine_lists" / "nudibranchs.txt").read_text().splitlines() if l.strip()]
win.cand_edit.setPlainText("\n".join(names))
win.cand_group = "nudibranchs"
win._update_cand_count()
win.log.appendPlainText(f"marine list: nudibranchs — {len(names):,} names")

# Normal working state: online sources on, a named place, a sensible search radius.
win.place.setText("Mabini, Batangas")
win.online_chk.setChecked(True)
win.lat.setText("13.715")
win.lon.setText("120.896")
win.radius.setValue(100)
win.place_label = "Mabini, PH, BT"
win._update_where()
idx = [i for i in range(win.group.count())
       if "nudibranch" in win.group.itemText(i).lower()]
if idx:
    win.group.setCurrentIndex(idx[0])
app.processEvents()

win.identify()
if not wait_for(lambda: win.worker is None):
    sys.exit("identification timed out")
app.processEvents()

print(f"  {win.cand_count.text()}")
for i in range(min(3, win.table.rowCount())):
    print(f"    #{i + 1} {win.table.item(i, 1).text()}  {win.table.item(i, 2).text()}")
if win.table.rowCount() == 0:
    sys.exit("no results — refusing to save a misleading screenshot")

# Show the Log tab: it carries the candidate source and the BioCLIP line.
for tw in win.findChildren(QTabWidget):
    for i in range(tw.count()):
        if tw.tabText(i).strip().lower() == "log":
            tw.setCurrentIndex(i)
for _ in range(6):
    app.processEvents()

pix = win.grab()
out.parent.mkdir(parents=True, exist_ok=True)
pix.save(str(out))
print(f"  saved {out} ({pix.width()}x{pix.height()})")
