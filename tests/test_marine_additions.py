"""Adding a species to the marine list must survive regenerating the lists.

The failure this pins: the group files are generated, so an append made by the app would vanish
on the next `make_marine_lists.py` run. Additions therefore live in `_user_added.txt`, which the
generator never writes, and every load unions it in.

Pure logic — no PyQt, no network, so this runs in CI without a display.
"""

import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import marine_additions as ma

fails = 0


def check(label, ok, detail: object = ""):
    global fails
    if ok:
        print(f"  ok   {label}")
    else:
        fails += 1
        print(f"  FAIL {label}   {str(detail)[:300]}")


print("1. names are cleaned, and junk is refused with a reason")
check("whitespace collapsed", ma.clean("  Chromodoris   annae \n") == "Chromodoris annae")
check("trailing punctuation dropped", ma.clean("Doto rosea,") == "Doto rosea")
check("a real binomial accepted",
      ma.reject_reason("Phyllodesmium poindimiei") is None,
      ma.reject_reason("Phyllodesmium poindimiei"))
check("hyphen and apostrophe allowed",
      ma.reject_reason("Cuthona nana-maris") is None and ma.reject_reason("Aplysia d'orbignyi") is None)
check("one word refused", "two words" in str(ma.reject_reason("nudibranch")),
      ma.reject_reason("nudibranch"))
check("camera filename refused", ma.reject_reason("OI3A2349") is not None)
check("placeholder refused", "placeholder" in str(ma.reject_reason("?")),
      ma.reject_reason("?"))
check("empty refused", ma.reject_reason("   ") is not None)
check("a sentence refused",
      ma.reject_reason("this is a nudibranch I saw at Lembeh") is not None,
      ma.reject_reason("this is a nudibranch I saw at Lembeh"))
check("a lowercase genus refused",
      ma.reject_reason("chromodoris annae") is not None,
      ma.reject_reason("chromodoris annae"))
check("a capitalised epithet refused",
      ma.reject_reason("Chromodoris Annae") is not None,
      ma.reject_reason("Chromodoris Annae"))
check("a rank qualifier refused",
      ma.reject_reason("Chromodoris sp.") is not None, ma.reject_reason("Chromodoris sp."))
check("a subspecies is allowed",
      ma.reject_reason("Chromodoris annae lutea") is None,
      ma.reject_reason("Chromodoris annae lutea"))

print("2. adding writes a named file with a header, once")
with tempfile.TemporaryDirectory() as td:
    path = ma.additions_path(td)
    check("the file name is the reserved one", path.name == "_user_added.txt", str(path))
    r = ma.add(path, "Chromodoris annae")
    check("first add succeeds", r["added"] is True, r)
    check("the file exists", path.is_file())
    text = path.read_text()
    check("a header explains what it is", text.startswith("#") and "regenerat" in text.lower(),
          text[:80])
    check("the name is in it", "Chromodoris annae" in text)

    r2 = ma.add(path, "chromodoris ANNAE")
    check("adding it again is refused, case-insensitively", r2["added"] is False, r2)
    check("and says why", "already in the file" in r2["reason"], r2["reason"])
    check("the file was not appended twice",
          path.read_text().count("Chromodoris annae") == 1,
          path.read_text().count("Chromodoris annae"))

print("3. blanks and comments in the file are ignored on read")
with tempfile.TemporaryDirectory() as td:
    path = ma.additions_path(td)
    path.write_text("# a comment\n\n  Doto rosea  \n# another\nDoto rosacea\nDoto rosea\n")
    names = ma.load(path)
    check("names in file order, no blanks or comments",
          names == ["Doto rosea", "Doto rosacea"], names)

print("4. merging keeps the original order and appends additions")
base = ["Chromodoris annae", "Doto rosea"]
extra = ["doto ROSEA", "Phyllodesmium poindimiei"]
merged = ma.merge(base, extra)
check("base order preserved", merged[:2] == base, merged)
check("only the new name is appended", merged == base + ["Phyllodesmium poindimiei"], merged)
check("case-insensitive match against the base", len(merged) == 3, merged)
check("merging nothing changes nothing", ma.merge(base, []) == base)

print("5. the additions SURVIVE regenerating the group lists")
with tempfile.TemporaryDirectory() as td:
    group = pathlib.Path(td) / "nudibranchs.txt"
    group.write_text("Chromodoris annae\nDoto rosea\n")
    ma.add(ma.additions_path(td), "Phyllodesmium poindimiei")
    before = ma.merge(group.read_text().split(), ma.load(ma.additions_path(td)))
    check ("the addition is in the merged list", "Phyllodesmium" in " ".join(before), before)

    # simulate `make_marine_lists.py` rewriting the group file from source
    group.write_text("Chromodoris annae\nDoto rosea\nDoto rosacea\n")
    after = ma.merge(group.read_text().split(), ma.load(ma.additions_path(td)))
    joined = " ".join(after)
    check("the new generated name appears", "Doto rosacea" in joined, after)
    check("the user's addition is still there", "Phyllodesmium" in joined, after)
    check("the additions file itself was untouched",
          ma.load(ma.additions_path(td)) == ["Phyllodesmium poindimiei"],
          ma.load(ma.additions_path(td)))

print("6. an unwritable location reports a reason instead of raising")
with tempfile.TemporaryDirectory() as td:
    blocked = pathlib.Path(td) / "ro"
    blocked.mkdir()
    target = blocked / "_user_added.txt"
    target.write_text("x\n")
    blocked.chmod(0o500)
    try:
        r = ma.add(target, "Doto rosea")
        check("either written or refused with a reason",
              r["added"] is True or "could not write" in r["reason"], r)
    finally:
        blocked.chmod(0o700)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
