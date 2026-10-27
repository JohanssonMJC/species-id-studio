"""Save a copy of an identified photo under a filename that says what it is.

Design rules, all of them consequences of things that went wrong in this project:

* **Copy, never move.** The original stays exactly where it was, with its original name.
  Anything that touches the user's photo library in place is not worth the risk.
* **Never overwrite.** If `Chromodoris_annae.jpg` is taken, the copy becomes
  `Chromodoris_annae_2.jpg`. Overwriting could destroy a different animal's photo that
  happened to be named the same way.
* **Never guess a name.** The caller passes the species; this module only makes it safe for
  a filesystem. If the caller has no species, it should not call.
* **Leave a manifest.** A filename cannot record which photo it came from, how confident the
  identification was, or which arm produced it — so every copy appends a line to
  `manifest.csv` beside it. Otherwise a folder of renamed copies is an unfalsifiable claim.
* **Verify after writing.** A short copy is a corrupt photo; the size is checked and the
  copy removed if it does not match.

Pure filesystem work, no Qt and no network, so CI can test it.
"""
from __future__ import annotations

import csv
import re
import shutil
import time
from pathlib import Path

# Everything that is not a letter, digit, dot, dash or underscore collapses to "_".
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

MANIFEST_NAME = "manifest.csv"
MANIFEST_HEADER = ["when", "copy", "source", "species", "confidence", "arm", "bytes"]


def safe_stem(species: str) -> str:
    """'Chromodoris annae' -> 'Chromodoris_annae'.

    Case is preserved deliberately: it keeps the genus capitalised, which is how a
    binomial is written and how the labelling rule reads names back.
    """
    s = " ".join(str(species or "").split())        # collapse whitespace
    s = _UNSAFE.sub("_", s).strip("._-")
    return s or "unidentified"


def default_folder(src: str | Path) -> Path:
    """Where copies go when no target is given: a sibling folder, so they stay together
    with the originals without mixing into them."""
    return Path(src).parent / "_identified"


def destination_for(src: str | Path, species: str, target_dir: str | Path | None = None,
                    suffix: str | None = None) -> Path:
    """The path a copy would take. Does not touch the disk beyond existence checks."""
    src = Path(src)
    ext = suffix or src.suffix or ".jpg"
    if not ext.startswith("."):
        ext = "." + ext
    folder = Path(target_dir) if target_dir else default_folder(src)
    stem = safe_stem(species)
    candidate = folder / f"{stem}{ext}"
    n = 2
    while candidate.exists():
        candidate = folder / f"{stem}_{n}{ext}"
        n += 1
        if n > 9999:                                # absurd guard against a loop
            raise RuntimeError(f"cannot find a free name for {stem}{ext} in {folder}")
    return candidate


def _already_saved(folder: Path, src: Path, species: str) -> Path | None:
    """Return the existing copy if this exact source was already saved as this species."""
    manifest = folder / MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        with manifest.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("source") == str(src.resolve())
                        and row.get("species") == species
                        and (folder / row.get("copy", "")).is_file()):
                    return folder / row["copy"]
    except Exception:
        return None
    return None


def save_copy(src: str | Path, species: str, *, target_dir: str | Path | None = None,
              suffix: str | None = None, manifest: bool = True,
              skip_if_saved: bool = True, confidence: str | None = None,
              arm: str | None = None) -> dict:
    """Copy `src` to a file named after `species`. Returns a small report dict."""
    src = Path(src)
    if not src.is_file():
        return {"saved": False, "reason": "source file not found", "source": str(src)}
    species = " ".join(str(species or "").split())
    if not species:
        return {"saved": False, "reason": "no species given", "source": str(src)}

    folder = Path(target_dir) if target_dir else default_folder(src)
    if skip_if_saved:
        existing = _already_saved(folder, src, species)
        if existing is not None:
            return {"saved": False, "reason": "already saved", "dest": str(existing),
                    "source": str(src), "species": species}

    dest = destination_for(src, species, target_dir, suffix)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)             # keeps EXIF: this is a real copy, not a re-encode
        if dest.stat().st_size != src.stat().st_size:
            dest.unlink(missing_ok=True)
            return {"saved": False, "reason": "copy was truncated", "source": str(src)}
    except Exception as exc:
        return {"saved": False, "reason": f"{type(exc).__name__}: {exc}",
                "source": str(src)}

    if manifest:
        try:
            write_manifest_row(folder, dest=dest, src=src, species=species,
                               confidence=confidence, arm=arm,
                               size=dest.stat().st_size)
        except Exception:
            pass                            # the copy matters more than the log line
    return {"saved": True, "reason": "copied", "dest": str(dest), "source": str(src),
            "species": species, "bytes": dest.stat().st_size}


def write_manifest_row(folder: str | Path, *, dest: Path, src: Path, species: str,
                       confidence: str | None = None, arm: str | None = None,
                       size: int | None = None) -> Path:
    """Append one provenance line. Writes the header if the file is new."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MANIFEST_NAME
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(MANIFEST_HEADER)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), dest.name, str(src.resolve()),
                    species, confidence or "", arm or "",
                    size if size is not None else ""])
    return path


def read_manifest(folder: str | Path) -> list[dict]:
    """Every row of the manifest, for auditing a folder of copies."""
    path = Path(folder) / MANIFEST_NAME
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
