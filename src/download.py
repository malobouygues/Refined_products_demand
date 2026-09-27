"""Fetch the raw inputs into data/. No source needs a key.

    python -m src.download            # everything
    python -m src.download fred cpb   # only these

Requests go through curl rather than urllib: FRED sits behind a CDN that hangs up on Python's
TLS handshake, while curl on the same URL answers in half a second.
"""

import io
import json
import re
import subprocess
import sys
import zipfile

import pandas as pd

from src import config as c

CURL = ["curl", "-sS", "-L", "--retry", "3", "--max-time", "900"]

# FRED id -> variable. One request per file: ids of mixed frequency come back as a zip.
FRED_US = {"DSPIC96": "inc", "TRFVOLUSM227NFWA": "vmt", "INDPRO": "ind", "AIRRPMTSI": "rpk",
           "POILBREUSDM": "brent", "CPIAUCSL": "cpi"}
FRED_CHINA = {"CRDQCNAPABIS": "lkq_credit"}  # BIS private non-financial credit, quarterly


def get(url):
    return subprocess.run(CURL + [url], capture_output=True, check=True).stdout


def jodi():
    """Monthly demand by product and country, 650 MB of csv that sql.py queries in place."""
    archive = get("https://www.jodidata.org/_resources/files/downloads/oil-data/"
                  "world_Secondary_CSV.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        (c.DATA / "jodi_secondary.csv").write_bytes(z.read(z.namelist()[0]))


def fred():
    for name, series in [("us", FRED_US), ("china", FRED_CHINA)]:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(series)
        df = pd.read_csv(io.BytesIO(get(url)), na_values=".")
        df.columns = ["date"] + [series[s] for s in df.columns[1:]]
        df.to_csv(c.DATA / f"fred_{name}.csv", index=False)


def ember():
    """Yearly electricity data for every country; sql.py keeps the China rows."""
    (c.DATA / "ember_yearly.csv").write_bytes(get(
        "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/"
        "yearly_full_release_long_format.csv"))


def worldbank():
    """China's gross capital formation in current USD, annual: the fallback for inv."""
    rows = json.loads(get("https://api.worldbank.org/v2/country/CHN/indicator/NE.GDI.TOTL.CD"
                          "?format=json&per_page=100"))[1]
    df = pd.DataFrame({"year": [r["date"] for r in rows], "inv": [r["value"] for r in rows]})
    df.dropna().sort_values("year").to_csv(c.DATA / "worldbank_china.csv", index=False)


def cpb():
    """World merchandise trade volume. The workbook name carries its vintage, so the link is
    read off the page; the one row used is kept as csv."""
    page = get("https://www.cpb.nl/en/worldtrademonitor/latest").decode("utf-8", "ignore")
    href = re.search(r'href="([^"]*world-trade-monitor[^"]*\.xlsx)"', page, re.I).group(1)
    book = pd.read_excel(io.BytesIO(get("https://www.cpb.nl" + href)), sheet_name="trade_out",
                         header=None)
    months = book.iloc[3, 5:].astype(str).str.replace("m", "-")
    # by series code: the label "World trade" is on the price row as well as the volume row
    volume = book[book[2] == "tgz_w1_qnmi_sn"].iloc[0, 5:].astype(float)
    df = pd.DataFrame({"month": months.to_numpy(), "wtr": volume.to_numpy()})
    df.to_csv(c.DATA / "cpb_world_trade.csv", index=False)


SOURCES = {"jodi": jodi, "fred": fred, "ember": ember, "worldbank": worldbank, "cpb": cpb}

if __name__ == "__main__":
    c.DATA.mkdir(exist_ok=True)
    for name in sys.argv[1:] or SOURCES:
        print(name)
        SOURCES[name]()
