#!/usr/bin/env python3
"""Pass 3: Semantic Scholar openAccessPdf, ats.aq page scraping, Cogitatio."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers"
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}

S2_DOIS = {
    "Agrawal2022": "10.1016/j.oneear.2022.05.014",
    "Kloppenburg2022": "10.1016/j.oneear.2022.02.004",
    "Kennicutt2019": "10.1016/j.oneear.2019.08.014",
    "Rolfer2025b": "10.1016/j.oneear.2025.101495",
    "Gardiner2025": "10.1111/cobi.14349",
    "Brooks2019RossSea": "10.1111/conl.12676",
    "Dodds2010": "10.1111/j.1758-5899.2009.00006.x",
    "Brady2011": "10.1080/2154896X.2011.626645",
    "Neffke2011": "10.1111/j.1944-8287.2011.01121.x",
    "Goldsworthy2022": "10.1007/s10784-021-09561-4",
    "Stone2013": "10.1007/s11558-013-9168-y",
    "Vabulas2013": "10.1007/s11558-012-9161-x",
    "Yao2021": "10.1177/13540661211033889",
    "Hidalgo2018": "10.1007/978-3-319-96661-8_46",
    "Guevara2016": "10.1007/s11192-016-2125-9",
    "Solomon2016": "10.1126/science.aae0061",
    "Haward2012": "10.1126/science.338.6107.603",
    "Sanchez2016": "10.1017/s0032247416000073",
    "Brady2017": "10.4337/9781784717681.00030",
    "Ocasio1997": "10.1002/(sici)1097-0266(199707)18:1+<187::aid-smj936>3.0.co;2-k",
    "BachrachBaratz1962": "10.2307/1952796",
    "Haas1992": "10.1017/S0020818300001442",
    "Steinberg2002": "10.1162/002081802320005504",
    "Koremenos2001": "10.1162/002081801317193592",
    "Young2011Effectiveness": "10.1073/pnas.1111690108",
    "HansenMagnusson2022": "10.17645/pag.v10i3.5332",
    "Meyer2022": "10.1126/science.adf3606",
    "Holland2020": "10.1126/science.aaz5491",
    "Chown2013": "10.1126/science.339.6116.141-a",
    "Turekian2025ScienceDiplomacy": "10.1126/science.aeb4815",
    "Wang2024Bioprospecting": "10.1016/j.marpol.2024.106338",
    "Boschma2017": "10.1080/00343404.2016.1254767",
    "Luo2024AntarcticTourism": "10.1038/d41586-024-02250-7",
    "Koivurova2005": "10.1017/s0731126500004923",
}

PAGE_SCRAPE = {
    "MadridProtocol1991": "https://www.ats.aq/e/protocol.html",
    "AntarcticTreaty1959": "https://www.ats.aq/e/antarctic_treaty.html",
    "ATCM48Outcomes2026": "https://www.ats.aq/devAS/Meetings/Past/101",
    "ATCMResolution4_2026": "https://www.ats.aq/devAS/Meetings/Measure/877",
    "HansenMagnusson2022b": "https://www.cogitatiopress.com/pag/article/view/5332",
}


def save(key: str, url: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
    except Exception as exc:
        return False, f"error: {exc}"
    if r.status_code != 200 or len(r.content) < 2000:
        return False, f"HTTP {r.status_code}, {len(r.content)}B"
    if r.content[:5] != b"%PDF-":
        return False, "not a PDF"
    path = OUT / f"{key}.pdf"
    path.write_bytes(r.content)
    return True, f"{path.name} ({len(r.content) // 1024} KB)"


def s2_oa(doi: str) -> str | None:
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            headers=UA,
            timeout=30,
        )
        if r.status_code != 200:
            return None
        oa = r.json().get("openAccessPdf")
        return oa.get("url") if oa else None
    except Exception:
        return None


def main() -> None:
    ok_count = 0
    for key, doi in S2_DOIS.items():
        if (OUT / f"{key}.pdf").exists():
            continue
        url = s2_oa(doi)
        if url:
            ok, msg = save(key, url)
            print(f"{key:30s} {'OK' if ok else 'FAIL':5s} {msg}", flush=True)
            if ok:
                ok_count += 1
        else:
            print(f"{key:30s} FAIL  no S2 OA pdf", flush=True)
        time.sleep(1.2)

    for key, page in PAGE_SCRAPE.items():
        real_key = key.rstrip("b") if key.endswith("2022b") else key
        if (OUT / f"{real_key}.pdf").exists():
            continue
        try:
            r = requests.get(page, headers=UA, timeout=60, allow_redirects=True)
        except Exception as exc:
            print(f"{real_key:30s} FAIL  page error {exc}", flush=True)
            continue
        links = re.findall(r'href=["\']([^"\']+\.pdf[^"\']*)', r.text, re.I)
        done = False
        for link in links[:5]:
            url = urljoin(r.url, link)
            ok, msg = save(real_key, url)
            print(f"{real_key:30s} {'OK' if ok else 'FAIL':5s} {msg} <- {url[:80]}", flush=True)
            if ok:
                ok_count += 1
                done = True
                break
            time.sleep(1)
        if not done and not links:
            print(f"{real_key:30s} FAIL  no pdf links on page", flush=True)

    print(f"\nPass 3: {ok_count} new PDFs")


if __name__ == "__main__":
    sys.exit(main())
