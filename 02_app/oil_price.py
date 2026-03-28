import requests
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()

EIA_SPOT_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
EIA_PRODUCTS = {
    "EPCWTI": "WTI",
    "EPCBRENT": "Brent",
}
PAGE_SIZE = 5000


def _get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv(name)


def _fetch_spot_price_page(api_key, offset):
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": PAGE_SIZE,
        "offset": offset,
    }

    for product in EIA_PRODUCTS:
        params.setdefault("facets[product][]", [])
        params["facets[product][]"].append(product)

    response = requests.get(EIA_SPOT_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("response", {})


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_oil_prices():
    key = _get_secret("EIA_API_KEY")
    if not key:
        return pd.DataFrame(columns=["WTI", "Brent"])

    all_rows = []
    offset = 0

    while True:
        payload = _fetch_spot_price_page(key, offset)
        rows = payload.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)

        total = payload.get("total")
        offset += len(rows)
        if total is not None and offset >= int(total):
            break

        if len(rows) < PAGE_SIZE:
            break

    if not all_rows:
        return pd.DataFrame(columns=["WTI", "Brent"])

    df = pd.DataFrame(all_rows)
    df = df[df["product"].isin(EIA_PRODUCTS)].copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period", "value"])

    price_table = (
        df.assign(series_name=df["product"].map(EIA_PRODUCTS))
        .pivot_table(index="period", columns="series_name", values="value", aggfunc="last")
        .sort_index()
    )

    return price_table[["WTI", "Brent"]].dropna(how="all")
