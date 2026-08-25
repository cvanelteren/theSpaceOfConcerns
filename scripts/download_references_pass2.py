#!/usr/bin/env python3
"""Second pass: scrape citation_pdf_url from DOI landing pages, try publisher
direct-PDF patterns, fix institutional document URLs."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers"
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}

WILEY_PDFDIRECT = {
    "Gardiner2025": "https://conbio.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/cobi.14349",
    "Brooks2019RossSea": "https://conbio.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/conl.12676",
    "Dodds2010": "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1758-5899.2009.00006.x",
    "Brady2011": "https://www.tandfonline.com/doi/pdf/10.1080/2154896X.2011.626645",
    "Neffke2011": "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1944-8287.2011.01121.x",
}

DIRECT_TRIES = {
    "AntarcticTreaty1959": ["https://www.ats.aq/documents/1959/ats1959.pdf"],
    "MadridProtocol1991": [
        "https://www.ats.aq/documents/recatt/Att006.pdf",
        "https://www.ats.aq/documents/recatt/att006.pdf",
    ],
    "HansenMagnusson2022": [
        "https://www.cogitatiopress.com/pag/article/download/5332/5332",
        "https://www.cogitatiopress.com/pag/article/view/5332",
    ],
    "UNGA1983Antarctica": [
        "https://undocs.org/pdf?symbol=en%2FA%2FRES%2F38%2F77",
    ],
    "AntarcticTreatyHandbook": [
        "https://www.ats.aq/documents/ATHandbook/ATHandbook.pdf",
        "https://www.ats.aq/docs/11154.pdf",
    ],
}

DOI_SCRAPE = [
    "Agrawal2022:10.1016/j.oneear.2022.05.014",
    "Kloppenburg2022:10.1016/j.oneear.2022.02.004",
    "Kennicutt2019:10.1016/j.oneear.2019.08.014",
    "Rolfer2025b:10.1016/j.oneear.2025.101495",
    "Moallemi2025a:10.1016/j.oneear.2025.101287",
    "Goldsworthy2022:10.1007/s10784-021-09561-4",
    "Stone2013:10.1007/s11558-013-9168-y",
    "Vabulas2013:10.1007/s11558-012-9161-x",
    "Orsini2019:10.1093/isr/viz005",
    "Yao2021:10.1177/13540661211033889",
    "Hidalgo2018:10.1007/978-3-319-96661-8_46",
    "Guevara2016:10.1007/s11192-016-2125-9",
    "Solomon2016:10.1126/science.aae0061",
    "Haward2012:10.1126/science.338.6107.603",
    "Dodds2010:10.1111/j.1758-5899.2009.00006.x",
    "Gardiner2025:10.1111/cobi.14349",
    "Brooks2019RossSea:10.1111/conl.12676",
    "Sanchez2016:10.1017/s0032247416000073",
    "Szpak2025:10.1017/S0954102025100278",
    "Drubel2023HiddenContestation:10.1017/S2045381722000259",
    "Brady2017:10.4337/9781784717681.00030",
    "Ocasio1997:10.1002/(sici)1097-0266(199707)18:1+<187::aid-smj936>3.0.co;2-k",
    "BachrachBaratz1962:10.2307/1952796",
    "Haas1992:10.1017/S0020818300001442",
    "Steinberg2002:10.1162/002081802320005504",
    "Koremenos2001:10.1162/002081801317193592",
    "Young2011Effectiveness:10.1073/pnas.1111690108",
    "Hidalgo2007:10.1126/science.1144581",
]


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


def scrape_pdf_meta(doi: str) -> str | None:
    try:
        r = requests.get(f"https://doi.org/{doi}", headers=UA, timeout=60, allow_redirects=True)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    m = re.search(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
        r.text,
        re.I,
    )
    if not m:
        m = re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url',
            r.text,
            re.I,
        )
    if m:
        pdf = m.group(1)
        if pdf.startswith("/"):
            from urllib.parse import urljoin

            pdf = urljoin(r.url, pdf)
        return pdf
    return None


def main() -> None:
    results = []

    for key, url in WILEY_PDFDIRECT.items():
        ok, msg = save(key, url)
        results.append((key, ok, msg))
        print(f"{key:30s} {'OK' if ok else 'FAIL':5s} {msg}", flush=True)
        time.sleep(1)

    for key, urls in DIRECT_TRIES.items():
        done = False
        for url in urls:
            ok, msg = save(key, url)
            print(f"{key:30s} {'OK' if ok else 'FAIL':5s} {msg} ({url})", flush=True)
            if ok:
                done = True
                break
            time.sleep(1)
        results.append((key, done, msg))

    for spec in DOI_SCRAPE:
        key, doi = spec.split(":", 1)
        if (OUT / f"{key}.pdf").exists():
            print(f"{key:30s} SKIP  already have PDF", flush=True)
            continue
        pdf = scrape_pdf_meta(doi)
        if pdf:
            ok, msg = save(key, pdf)
            print(f"{key:30s} {'OK' if ok else 'FAIL':5s} {msg}", flush=True)
            results.append((key, ok, msg))
        else:
            print(f"{key:30s} FAIL  no citation_pdf_url meta", flush=True)
            results.append((key, False, "no meta tag"))
        time.sleep(1)

    ok = sum(1 for _, o, _ in results if o)
    print(f"\nPass 2: {ok}/{len(results)} new PDFs")


if __name__ == "__main__":
    sys.exit(main())
