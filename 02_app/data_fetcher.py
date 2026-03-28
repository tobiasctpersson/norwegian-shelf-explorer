import pandas as pd

SODIR_URL = "https://factpages.sodir.no/en/wellbore/tableview/exploration/all"


def _coerce_decimal_coordinate(series):
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")

def load_wellbore_data():
    df = pd.read_html(SODIR_URL)[0]
    df['Entered date'] = pd.to_datetime(df['Entered date'], dayfirst=True)
    df['NS decimal degrees'] = _coerce_decimal_coordinate(df['NS decimal degrees'])
    df['EW decimal degrees'] = _coerce_decimal_coordinate(df['EW decimal degrees'])
    df['Year'] = df['Entered date'].dt.year
    return df

def get_summary_stats(df):
    total = len(df)
    pa = len(df[df['Status'] == 'P&A'])
    dry = len(df[df['Content'] == 'DRY'])
    operators = df['Drilling operator'].nunique()
    year_min = int(df['Year'].min())
    year_max = int(df['Year'].max())

    return {
        "total_wells": total,
        "pa_wells": pa,
        "pa_pct": round(pa / total * 100, 1),
        "dry_wells": dry,
        "unique_operators": operators,
        "year_range": f"{year_min}–{year_max}"
    }
