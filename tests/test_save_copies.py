"""Copying an identified photo under its species name. Filesystem only, so CI runs it."""
import csv
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import save_copies as sc

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  ok   {name}")
    else:
        fails += 1
        print(f"  FAIL {name}  {detail}")


print("1. species names become safe filenames without losing the binomial")
for species, want in (("Chromodoris annae", "Chromodoris_annae"),
                      ("  Nembrotha   chamberlaini ", "Nembrotha_chamberlaini"),
                      ("Antennarius maculatus (Desmarest, 1817)", "Antennarius_maculatus_Desmarest_1817"),
                      ("Genus sp.", "Genus_sp"),
                      ("../../etc/passwd", "etc_passwd"),
                      ("", "unidentified"),
                      ("a/b\\c:d*e?f", "a_b_c_d_e_f")):
    got = sc.safe_stem(species)
    check(f"{species!r:44s} -> {got!r}", got == want, f"want {want!r}")

print("2. a copy lands in a sibling _identified folder, named after the species")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    src = root / "OI3A2349.jpg"
    src.write_bytes(b"pretend this is a photo" * 100)
    r = sc.save_copy(src, "Chromodoris annae")
    check("saved", r["saved"], r)
    check("dest is in _identified", r["dest"] == str(root / "_identified" / "Chromodoris_annae.jpg"),
          r["dest"])
    check("copy exists", pathlib.Path(r["dest"]).is_file())
    check("copy is byte-identical", pathlib.Path(r["dest"]).read_bytes() == src.read_bytes())
    check("ORIGINAL still there, untouched", src.is_file() and src.name == "OI3A2349.jpg")

print("3. it never overwrites: a second animal gets _2, _3")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    a, b = root / "a.jpg", root / "b.jpg"
    a.write_bytes(b"A" * 50)
    b.write_bytes(b"B" * 60)
    r1 = sc.save_copy(a, "Chromodoris annae", skip_if_saved=False)
    r2 = sc.save_copy(b, "Chromodoris annae", skip_if_saved=False)
    r3 = sc.save_copy(a, "Chromodoris annae", skip_if_saved=False)
    names = [pathlib.Path(x["dest"]).name for x in (r1, r2, r3)]
    check("three distinct files", len(set(names)) == 3, names)
    check("first keeps the plain name", names[0] == "Chromodoris_annae.jpg", names)
    check("second is suffixed", names[1] == "Chromodoris_annae_2.jpg", names)
    check("third is suffixed again", names[2] == "Chromodoris_annae_3.jpg", names)
    check("no data lost", pathlib.Path(r2["dest"]).read_bytes() == b.read_bytes())

print("4. re-saving the same photo as the same species is skipped, not duplicated")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    src = root / "x.jpg"
    src.write_bytes(b"photo" * 20)
    first = sc.save_copy(src, "Thor amboinensis")
    second = sc.save_copy(src, "Thor amboinensis")
    check("first saved", first["saved"])
    check("second skipped", not second["saved"] and second["reason"] == "already saved", second)
    check("only one file in the folder",
          len(list((root / "_identified").glob("*.jpg"))) == 1,
          list((root / "_identified").iterdir()))
    # a DIFFERENT species for the same photo IS saved (a correction)
    third = sc.save_copy(src, "Thor spinosus")
    check("a corrected species is a new copy", third["saved"], third)

print("5. every copy leaves a manifest that can be audited")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    src = root / "y.jpg"
    src.write_bytes(b"data" * 30)
    sc.save_copy(src, "Pteraeolidia semperi", )
    sc.write_manifest_row(root / "_identified", dest=root / "_identified" / "note.jpg",
                          src=src, species="Pteraeolidia semperi",
                          confidence="confident", arm="your labels")
    rows = sc.read_manifest(root / "_identified")
    check("rows recorded", len(rows) == 2, rows)
    check("source path recorded", rows[0]["source"] == str(src.resolve()), rows[0])
    check("species recorded", rows[0]["species"] == "Pteraeolidia semperi", rows[0])
    check("arm recorded when given", rows[1]["arm"] == "your labels", rows[1])
    check("header written once",
          (root / "_identified" / "manifest.csv").read_text().count("when,copy,source") == 1)

print("5b. confidence and arm land in the SAME row, not a second blank one")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    src = root / "v.jpg"
    src.write_bytes(b"v" * 40)
    sc.save_copy(src, "Chromodoris annae", confidence="confident", arm="your labels")
    rows = sc.read_manifest(root / "_identified")
    check("exactly one row", len(rows) == 1, rows)
    check("confidence recorded", rows[0]["confidence"] == "confident", rows[0])
    check("arm recorded", rows[0]["arm"] == "your labels", rows[0])

print("6. it refuses rather than guesses")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    r = sc.save_copy(root / "missing.jpg", "Chromodoris annae")
    check("missing source is reported", not r["saved"] and "not found" in r["reason"], r)
    src = root / "z.jpg"
    src.write_bytes(b"x")
    r2 = sc.save_copy(src, "   ")
    check("empty species is refused", not r2["saved"] and "species" in r2["reason"], r2)
    check("nothing was created for a refusal",
          not (root / "_identified").exists(), list(root.iterdir()))

print("7. a custom target folder is honoured, and the suffix can be changed")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    src = root / "w.png"
    src.write_bytes(b"\x89PNG" + b"0" * 40)
    target = root / "sorted"
    r = sc.save_copy(src, "Amphiprion ocellaris", target_dir=target)
    check("custom dir used", pathlib.Path(r["dest"]).parent == target, r["dest"])
    check("extension preserved", r["dest"].endswith(".png"), r["dest"])
    r2 = sc.save_copy(src, "Amphiprion clarkii", target_dir=target, suffix=".jpg")
    check("suffix override honoured", r2["dest"].endswith("Amphiprion_clarkii.jpg"), r2["dest"])

print("8. destination_for() is a pure planning call")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    src = root / "q.jpg"
    src.write_bytes(b"q")
    plan = sc.destination_for(src, "Chromodoris annae")
    check("planned path is not created", not plan.exists(), plan)
    check("planned path matches what save_copy writes", plan.name == "Chromodoris_annae.jpg")

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
