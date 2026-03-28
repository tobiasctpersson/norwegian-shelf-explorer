import pandas as pd
import requests
import streamlit as st

SODIR_URL = "https://factpages.sodir.no/en/wellbore/tableview/exploration/all"
FIELD_OUTLINES_URL = "https://factmaps.sodir.no/api/rest/services/DataService/Data/MapServer/7100/query"


def _coerce_decimal_coordinate(series):
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def _ring_area(ring):
    if len(ring) < 3:
        return 0.0

    area = 0.0
    for idx in range(len(ring)):
        x1, y1 = ring[idx]
        x2, y2 = ring[(idx + 1) % len(ring)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _ring_centroid(ring):
    if len(ring) < 3:
        if not ring:
            return None, None
        lon_values = [point[0] for point in ring]
        lat_values = [point[1] for point in ring]
        return sum(lon_values) / len(lon_values), sum(lat_values) / len(lat_values)

    area = _ring_area(ring)
    if abs(area) < 1e-12:
        lon_values = [point[0] for point in ring]
        lat_values = [point[1] for point in ring]
        return sum(lon_values) / len(lon_values), sum(lat_values) / len(lat_values)

    cx = 0.0
    cy = 0.0
    for idx in range(len(ring)):
        x1, y1 = ring[idx]
        x2, y2 = ring[(idx + 1) % len(ring)]
        factor = x1 * y2 - x2 * y1
        cx += (x1 + x2) * factor
        cy += (y1 + y2) * factor

    factor = 1 / (6 * area)
    return cx * factor, cy * factor


def _extract_outer_rings(geometry):
    if not geometry:
        return []

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        return [coordinates[0]] if coordinates else []

    if geometry_type == "MultiPolygon":
        return [polygon[0] for polygon in coordinates if polygon]

    return []


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_wellbore_data():
    df = pd.read_html(SODIR_URL)[0]
    df['Entered date'] = pd.to_datetime(df['Entered date'], dayfirst=True)
    df['NS decimal degrees'] = _coerce_decimal_coordinate(df['NS decimal degrees'])
    df['EW decimal degrees'] = _coerce_decimal_coordinate(df['EW decimal degrees'])
    df['Year'] = df['Entered date'].dt.year
    return df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_field_outlines():
    params = {
        "where": "1=1",
        "outFields": "fldName,fldNpdidField,fldMainArea,fldCurrentActivitySatus",
        "returnGeometry": "true",
        "f": "geojson",
    }

    response = requests.get(FIELD_OUTLINES_URL, params=params, timeout=45)
    response.raise_for_status()
    geojson = response.json()
    features = geojson.get("features", [])

    field_records = []
    filtered_features = []
    for feature in features:
        properties = feature.get("properties", {})
        field_id = properties.get("fldNpdidField")
        rings = _extract_outer_rings(feature.get("geometry"))
        if not rings or field_id is None:
            continue

        ring_areas = [abs(_ring_area(ring)) for ring in rings]
        largest_ring = rings[ring_areas.index(max(ring_areas))]
        label_lon, label_lat = _ring_centroid(largest_ring)
        field_name = properties.get("fldName") or "Unnamed field"
        main_area = properties.get("fldMainArea") or "Unknown"
        status = properties.get("fldCurrentActivitySatus") or "Unknown"
        feature["properties"]["fldNpdidField"] = str(field_id)
        feature["properties"]["name"] = field_name
        feature["properties"]["kind"] = "Field"
        feature["properties"]["field_id"] = str(field_id)
        feature["properties"]["main_area"] = main_area
        feature["properties"]["status"] = status
        filtered_features.append(feature)

        field_records.append(
            {
                "field_name": field_name,
                "field_id": str(field_id),
                "main_area": main_area,
                "status": status,
                "label_lon": label_lon,
                "label_lat": label_lat,
                "rings": rings,
            }
        )

    return {
        "geojson": {
            "type": "FeatureCollection",
            "features": filtered_features,
        },
        "labels": field_records,
        "outlines": field_records,
    }

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
