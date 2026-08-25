#!/usr/bin/env python3
"""Download open-access versions of all cited references into papers/.

Routes:
- DOI entries: Unpaywall API (legal OA lookup); download best OA PDF.
- Zenodo DOIs: Zenodo API.
- URL-only entries: direct download (institutional documents).
- UNGA resolution: undocs.org.
Paywalled items are reported, not bypassed.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "theSpaceOfConcerns.bib"
OUT = ROOT / "papers"
REPORT = OUT / "DOWNLOAD_REPORT.md"
EMAIL = "unpaywall-contact@example.org"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) reference-downloader/1.0"}

EXTRA_DOIS = {"Hidalgo2018": "10.1007/978-3-319-96661-8_46"}
SKIP = {"vanElteren2025": "software repository (GitHub), not a paper"}


def parse_bib() -> dict[str, dict[str, str]]:
    bib = BIB.read_text(encoding="utf-8")
    entries = {}
    for m in re.finditer(r"@\w+\s*\{", bib):
        j = m.end()
        key = ""
        while j < len(bib) and bib[j] not in ",\n":
            key += bib[j]
            j += 1
        key = key.strip()
        depth = 1
        j += 1
        while j < len(bib) and depth > 0:
            if bib[j] == "{":
                depth += 1
            elif bib[j] == "}":
                depth -= 1
            j += 1
        body = bib[m.end() + len(key) + 1 : j - 1]
        fields = {}
        doi = re.search(r"doi\s*=\s*\{(.*?)\}", body, re.S)
        url = re.search(r"\burl\s*=\s*\{(.*?)\}", body, re.S)
        if doi:
            fields["doi"] = doi.group(1).strip()
        if url:
            fields["url"] = url.group(1).strip()
        entries[key] = fields
    return entries


def save(key: str, url: str, timeout: int = 90) -> tuple[bool, str]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
    except Exception as exc:
        return False, f"download error: {exc}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    ctype = r.headers.get("Content-Type", "")
    head = r.content[:5]
    if head == b"%PDF-" or "pdf" in ctype:
        ext = ".pdf"
    elif "<html" in r.content[:500].decode("utf-8", "ignore").lower():
        ext = ".html"
    else:
        ext = ".bin"
    path = OUT / f"{key}{ext}"
    path.write_bytes(r.content)
    return True, f"{path.name} ({len(r.content) // 1024} KB)"


def try_unpaywall(key: str, doi: str) -> tuple[bool, str]:
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": EMAIL},
            headers=UA,
            timeout=30,
        )
        if r.status_code != 200:
            return False, f"unpaywall HTTP {r.status_code}"
        data = r.json()
    except Exception as exc:
        return False, f"unpaywall error: {exc}"
    if not data.get("is_oa"):
        return False, "paywalled (no OA version)"
    locs = []
    if data.get("best_oa_location"):
        locs.append(data["best_oa_location"])
    locs.extend(data.get("oa_locations", []))
    for loc in locs:
        pdf = loc.get("url_for_pdf")
        if pdf:
            ok, msg = save(key, pdf)
            if ok:
                return True, msg
    for loc in locs:
        page = loc.get("url")
        if page:
            ok, msg = save(key, page)
            if ok:
                return True, msg + " [landing page, not PDF]"
    return False, "OA flagged but no working URL"


def try_zenodo(key: str, doi: str) -> tuple[bool, str]:
    rec = doi.split("zenodo.")[-1]
    try:
        r = requests.get(f"https://zenodo.org/api/records/{rec}", headers=UA, timeout=30)
        if r.status_code != 200:
            return False, f"zenodo HTTP {r.status_code}"
        files = r.json().get("files", [])
        if not files:
            return False, "zenodo record has no files"
        best = max(files, key=lambda f: f.get("size", 0))
        ok, msg = save(key, best["links"]["self"], timeout=300)
        return ok, msg
    except Exception as exc:
        return False, f"zenodo error: {exc}"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    entries = parse_bib()
    results: list[tuple[str, str, str]] = []

    for key, fields in entries.items():
        if key in SKIP:
            results.append((key, "SKIPPED", SKIP[key]))
            continue
        doi = fields.get("doi") or EXTRA_DOIS.get(key)
        url = fields.get("url")
        status = detail = ""

        if key == "UNGA1983Antarctica":
            ok, detail = save(key, "https://undocs.org/pdf?symbol=en/A/RES/38/77")
            status = "OK" if ok else "FAILED"
        elif doi and doi.startswith("10.5281/zenodo"):
            ok, detail = try_zenodo(key, doi)
            status = "OK" if ok else "FAILED"
            if not ok and doi:
                ok2, d2 = try_unpaywall(key, doi)
                if ok2:
                    status, detail = "OK", d2
        elif doi:
            ok, detail = try_unpaywall(key, doi)
            status = "OK" if ok else "FAILED"
            if not ok and url:
                ok2, d2 = save(key, url)
                if ok2:
                    status, detail = "OK", d2 + " [bib URL]"
        elif url:
            ok, detail = save(key, url)
            status = "OK" if ok else "FAILED"
        else:
            status, detail = "FAILED", "no DOI or URL (book/chapter)"

        results.append((key, status, detail))
        print(f"{key:30s} {status:8s} {detail}", flush=True)
        time.sleep(0.5)

    ok = [r for r in results if r[1] == "OK"]
    failed = [r for r in results if r[1] == "FAILED"]
    skipped = [r for r in results if r[1] == "SKIPPED"]

    lines = [
        "# Reference download report",
        "",
        f"- Downloaded: {len(ok)}/{len(results)}",
        f"- Failed (paywalled or no source): {len(failed)}",
        f"- Skipped: {len(skipped)}",
        "",
        "## Downloaded",
        "",
    ]
    lines += [f"- `{k}`: {d}" for k, _, d in ok]
    lines += ["", "## Not downloaded", ""]
    lines += [f"- `{k}`: {d}" for k, _, d in failed]
    lines += ["", "## Skipped", ""]
    lines += [f"- `{k}`: {d}" for k, _, d in skipped]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone: {len(ok)} downloaded, {len(failed)} failed, {len(skipped)} skipped")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
