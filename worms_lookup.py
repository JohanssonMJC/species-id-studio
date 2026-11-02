"""Resolve a species name to its WoRMS page through the REST API.

Why not a link: WoRMS's `aphia.php?p=taxlist&searchpar=<name>` is a SEARCH page, and it ignores
the parameter — the browser opens the WoRMS front page with no species selected, which is what
"opens worms but not for the species" looked like. The REST API answers with the AphiaID, and
`p=taxdetails&id=<AphiaID>` is the species page itself.

Pure standard library so the parsing rules can be tested without Qt or a network.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.marinespecies.org/rest/AphiaRecordsByName/"
SEARCH = "https://www.marinespecies.org/aphia.php?p=taxlist&tName="
DETAILS = "https://www.marinespecies.org/aphia.php?p=taxdetails&id="
UA = "species-id-studio/1.0 (WoRMS lookup)"


def detail_url(aphia_id: int | str) -> str:
    """The species page for an AphiaID — the thing the link was supposed to open."""
    return f"{DETAILS}{int(aphia_id)}"


def search_url(name: str) -> str:
    """The search page, for when the name resolves to nothing. A fallback, not the answer."""
    return SEARCH + urllib.parse.quote(str(name or "").strip())


def parse_records(payload) -> list[dict]:
    """Turn the REST response into the few fields a reader needs, in API order.

    Accepts the decoded JSON or the raw text. A 404 body ("No such record") is not an error:
    it means WoRMS does not know the name, which is a usable answer.
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "replace")
    if isinstance(payload, str):
        text = payload.strip()
        if not text or text.lstrip().lower().startswith("<"):
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, list):
        # the API returns a bare object for a single hit in some versions
        payload = [payload] if isinstance(payload, dict) and payload.get("AphiaID") else []
    out: list[dict] = []
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        aid = rec.get("AphiaID")
        if not aid:
            continue
        out.append({
            "aphia_id": int(aid),
            "name": (rec.get("scientificname") or "").strip(),
            "authority": (rec.get("authority") or "").strip(),
            "status": (rec.get("status") or "").strip(),
            "rank": (rec.get("rank") or "").strip(),
            "url": rec.get("url") or detail_url(aid),
        })
    return out


def fetch(name: str, timeout: float = 20.0, marine_only: bool = True) -> list[dict]:
    """Ask WoRMS what it knows about this name. Network errors raise; empty means "unknown"."""
    n = str(name or "").strip()
    if not n:
        return []
    query = urllib.parse.urlencode({"like": "false", "marine_only": "true" if marine_only else "false"})
    url = f"{API}{urllib.parse.quote(n)}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return parse_records(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:          # "No such record" — a legitimate empty answer
            return []
        raise


def choose(records: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick an unambiguous record, or return all of them for the user to choose.

    Never silently choose between several: a name can be an accepted species and a synonym of
    something else, or reused in two phyla. `accepted` is preferred only when it is unique.
    """
    if not records:
        return None, []
    accepted = [r for r in records if r["status"].lower() == "accepted"]
    if len(accepted) == 1:
        return accepted[0], records
    if len(records) == 1:
        return records[0], records
    return None, records


def label(rec: dict) -> str:
    """One line per record for a picker: name, authority, status, rank, AphiaID."""
    bits = [rec["name"]]
    if rec["authority"]:
        bits.append(rec["authority"])
    line = " ".join(bits)
    tail = [x for x in (rec["status"], rec["rank"]) if x]
    if tail:
        line += f" — {', '.join(tail)}"
    return f"{line}  (AphiaID {rec['aphia_id']})"
