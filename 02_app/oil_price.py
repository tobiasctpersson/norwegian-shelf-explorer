import requests
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def _get_secret(name):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)

def load_oil_prices():
    key = _get_secret("EIA_API_KEY")
    if not key:
        return pd.DataFrame(columns=["WTI", "Brent"])
    
    url = (
        f"https://api.eia.gov/v2/petroleum/pri/spt/data/"
        f"?api_key={key}"
        f"&frequency=monthly"
        f"&data[0]=value"
        f"&facets[product][]=EPCBRENT"
        f"&facets[product][]=EPCWTI"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=asc"
        f"&length=500"
        f"&offset=0"
    )
    
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()['response']['data']
    df = pd.DataFrame(data)
    df['period'] = pd.to_datetime(df['period'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    wti = df[df['product'] == 'EPCWTI'][['period', 'value']].rename(columns={'value': 'WTI'}).set_index('period')
    brent = df[df['product'] == 'EPCBRENT'][['period', 'value']].rename(columns={'value': 'Brent'}).set_index('period')
    
    result = pd.concat([wti, brent], axis=1).dropna(how='all')
    return result
