"""WoRMS name resolution: the parse rules, the ambiguity rule, and one live call.

Pure logic runs without a network; the live check at the end is skipped when the API is
unreachable, so CI never depends on marinespecies.org being up.
"""

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import worms_lookup as wl

fails = 0


def check(label, ok, detail: object = ""):
    global fails
    if ok:
        print(f"  ok   {label}")
    else:
        fails += 1
        print(f"  FAIL {label}   {str(detail)[:300]}")


print("1. a resolved name becomes a SPECIES page, not a search page")
check("detail url carries the AphiaID", wl.detail_url(558229).endswith("taxdetails&id=558229"),
      wl.detail_url(558229))
check("a string AphiaID is tolerated", wl.detail_url("558229").endswith("id=558229"))
check("the search url is only a fallback and encodes the name",
      wl.search_url("Chromodoris annae").endswith("tName=Chromodoris%20annae"), wl.search_url("X"))

print("2. the REST payload is parsed into what a reader needs")
payload = '''[{"AphiaID":558229,"url":"https://www.marinespecies.org/aphia.php?p=taxdetails&id=558229",
"scientificname":"Chromodoris annae","authority":"Bergh, 1877","status":"accepted","rank":"Species"}]'''
recs = wl.parse_records(payload)
check("one record", len(recs) == 1, recs)
check("aphia id", recs[0]["aphia_id"] == 558229, recs[0])
check("name and authority", recs[0]["name"] == "Chromodoris annae"
      and recs[0]["authority"] == "Bergh, 1877", recs[0])
check("status and rank kept", recs[0]["status"] == "accepted" and recs[0]["rank"] == "Species")
check("url taken from the payload", recs[0]["url"].endswith("id=558229"))

print("3. empty and malformed bodies are empty answers, not crashes")
for body in ("", "[]", "No such record", "<html>503</html>", "{not json", "null"):
    check(f"empty for {body!r}", wl.parse_records(body) == [], wl.parse_records(body))
check("a bare object still parses", len(wl.parse_records(
    '{"AphiaID":1,"scientificname":"Alpha beta","status":"accepted"}')) == 1)
check("a record with no AphiaID is dropped", wl.parse_records('[{"scientificname":"x"}]') == [])
check("a missing url is rebuilt", wl.parse_records(
    '[{"AphiaID":42,"scientificname":"Alpha beta"}]')[0]["url"].endswith("id=42"))

print("4. an unambiguous accepted record is opened directly")
one = wl.parse_records(payload)
pick, all_ = wl.choose(one)
check("chosen", pick is not None and pick["aphia_id"] == 558229, pick)
check("all records still returned", len(all_) == 1)

print("5. a synonym does not win over the accepted name")
syn = wl.parse_records('''[
 {"AphiaID":10,"scientificname":"Alpha beta","status":"unaccepted","rank":"Species"},
 {"AphiaID":11,"scientificname":"Alpha beta","status":"accepted","rank":"Species"}]''')
pick, _ = wl.choose(syn)
check("the accepted record is opened", pick is not None and pick["aphia_id"] == 11, pick)

print("6. two accepted records are NOT silently resolved — the user picks")
amb = wl.parse_records('''[
 {"AphiaID":20,"scientificname":"Alpha beta","status":"accepted","rank":"Species"},
 {"AphiaID":21,"scientificname":"Alpha beta","status":"accepted","rank":"Species"}]''')
pick, all_ = wl.choose(amb)
check("nothing is chosen", pick is None, pick)
check("both are offered", len(all_) == 2, all_)

print("7. nothing found is a usable answer")
pick, all_ = wl.choose([])
check("no pick, no records", pick is None and all_ == [])

print("8. picker labels identify the record")
lab = wl.label({"aphia_id": 558229, "name": "Chromodoris annae", "authority": "Bergh, 1877",
                "status": "accepted", "rank": "Species", "url": ""})
check("name, authority, status, rank, id", "Chromodoris annae Bergh, 1877" in lab
      and "accepted" in lab and "Species" in lab and "558229" in lab, lab)

print("9. one live lookup, when the API is reachable")
try:
    live = wl.fetch("Chromodoris annae")
except Exception as exc:
    print(f"  skip network check ({type(exc).__name__})")
else:
    check("resolved", bool(live) and live[0]["aphia_id"] == 558229, live[:1])
    check("accepted and a species", live[0]["status"] == "accepted" and live[0]["rank"] == "Species",
          live[0])
    check("the url is the species page", live[0]["url"].endswith("id=558229"), live[0]["url"])
    unknown = wl.fetch("Nonexistus fakus")
    check("an invented name returns nothing", unknown == [], unknown)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
