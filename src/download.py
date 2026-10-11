"""Fetch the raw inputs into data/, one folder per product. No source needs a key.

    python -m src.download            # everything
    python -m src.download eia mpa    # only these

Requests go through curl rather than urllib: FRED sits behind a CDN that hangs up on Python's
TLS handshake, while curl on the same URL answers in half a second.
"""

import io
import json
import re
import subprocess
import sys
import time

import pandas as pd

from src import config as c

CURL = ["curl", "-sS", "-L", "--retry", "3", "--max-time", "900"]

# EIA id -> column, per product and frequency. Product supplied and exports in kb/d,
# stocks in thousand barrels, retail price in USD/gal. The monthly product supplied is the target;
# the weekly one is the same concept a week after the fact, so it covers the months the monthly
# has not reached yet. Same column names for both products, so the code downstream is shared.
EIA = {c.GASOLINE: {"monthly": {"MGFUPUS2": "product_supplied", "MGFEXUS2": "exports",
                                "EMM_EPMR_PTE_NUS_DPG": "retail_price"},
                    "weekly": {"WGFUPUS2": "product_supplied", "WGTSTUS1": "stocks"}},
       c.DISTILLATE: {"monthly": {"MDIUPUS2": "product_supplied", "MDIEXUS2": "exports",
                                  "EMD_EPD2D_PTE_NUS_DPG": "retail_price"},
                      "weekly": {"WDIUPUS2": "product_supplied", "WDISTUS1": "stocks"}}}

# FRED id -> column, per product, all monthly: one request each, since mixed frequencies come
# back as a zip. The CPI deflates the retail prices.
FRED = {c.GASOLINE: {"TRFVOLUSM227NFWA": "vmt", "PAYEMS": "payrolls", "DSPIC96": "inc",
                     "CPIAUCSL": "cpi"},
        c.DISTILLATE: {"INDPRO": "ind", "IPMAN": "ind_manuf", "TRUCKD11": "truck",
                       "CPIAUCSL": "cpi"},
        c.BUNKER: {"POILBREUSDM": "brent"}}

NOAA = ("https://ftp.cpc.ncep.noaa.gov/htdocs/products/analysis_monitoring/cdus/degree_days/"
        "archives/Heating%20degree%20Days/monthly%20states/")

# MPA datasets on data.gov.sg, monthly: sales by fuel grade (kt), vessel calls by purpose,
# total cargo throughput (kt), tanker arrivals
MPA = {"bunker_sales": "d_4f5abbf4486bf8e52bbed3be56dde562",
       "vessel_calls": "d_48cb38d12697d3463c8cadfb22e6c61d",
       "cargo": "d_042dd8b935eab998f389adaf559c80de",
       "tankers": "d_9adb5ace517591edd9a8c88291ac1f1c"}


def get(url):
    return subprocess.run(CURL + [url], capture_output=True, check=True).stdout


def eia():
    """EIA history files are .xls with one series each, named <id>m or <id>w, dated mid-month or
    on the Friday that ends the week."""
    for folder, frequencies in EIA.items():
        for freq, series in frequencies.items():
            columns = []
            for code, column in series.items():
                url = f"https://www.eia.gov/dnav/pet/hist_xls/{code}{freq[0]}.xls"
                book = pd.read_excel(io.BytesIO(get(url)), sheet_name="Data 1", header=2)
                columns.append(book.set_axis(["date", column], axis=1).set_index("date")[column])
            pd.concat(columns, axis=1, sort=True).to_csv(folder / f"eia_{freq}.csv")


def fred():
    for folder, series in FRED.items():
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(series)
        df = pd.read_csv(io.BytesIO(get(url)), na_values=".")
        df.columns = ["date"] + [series[s] for s in df.columns[1:]]
        df.to_csv(folder / "fred.csv", index=False)


def noaa():
    """US heating degree days, weighted by population and by homes heated with fuel oil. One
    text file per month since mid-1997; old file names are irregular, so the month is read off
    the header. The national lines come in a fixed order: population, gas, oil, electric."""
    rows = []
    for year in re.findall(r'href="(\d{4})/"', get(NOAA).decode()):
        for name in re.findall(r'href="([^"]+\.txt)"', get(NOAA + year + "/").decode()):
            text = get(NOAA + year + "/" + name).decode("latin-1")
            us = [int(line.split()[2]) for line in text.splitlines()
                  if line.strip().startswith("UNITED STATES")]
            month = re.search(r"MONTHLY DATA FOR (\w{3} \d{4})", text).group(1)
            rows.append({"month": month, "hdd": us[0], "hdd_oil": us[2]})
    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month"], format="%b %Y").dt.strftime("%Y-%m")
    df = df.drop_duplicates("month").sort_values("month")
    df.to_csv(c.DISTILLATE / "noaa_hdd.csv", index=False)


def mpa():
    """Singapore port statistics, kept as published. data.gov.sg serves each csv through a
    signed link once prepared, and without a key it allows a few calls a minute."""
    for name, dataset in MPA.items():
        api = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset}/"
        get(api + "initiate-download")
        for _ in range(10):
            time.sleep(20)
            poll = json.loads(get(api + "poll-download"))["data"] or {}  # None when rate-limited
            if "url" in poll:
                break
        (c.BUNKER / f"mpa_{name}.csv").write_bytes(get(poll["url"]))


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
    df.to_csv(c.BUNKER / "cpb_world_trade.csv", index=False)


SOURCES = {"eia": eia, "fred": fred, "noaa": noaa, "mpa": mpa, "cpb": cpb}

if __name__ == "__main__":
    for folder in (c.GASOLINE, c.DISTILLATE, c.BUNKER):
        folder.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or SOURCES:
        print(name)
        SOURCES[name]()
