import base64
import html
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_fetcher import get_summary_stats, load_field_outlines, load_wellbore_data
from llm_query import ask_claude
from oil_price import load_oil_prices

st.set_page_config(page_title="Norwegian Shelf Explorer", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=DM+Mono:wght@300;400&display=swap');

:root {
  --bg: #0a0a0f;
  --bg2: #0f0f16;
  --surface: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.08);
  --border-bright: rgba(255,255,255,0.14);
  --text: #e8e8f0;
  --text-muted: #7a7a96;
  --text-dim: #4a4a60;
  --accent: #c8d4f0;
  --mono: 'DM Mono', monospace;
  --sans: 'Sora', sans-serif;
}

html, body, [class*="css"] {
  background-color: #0a0a0f !important;
  color: #e8e8f0 !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 300;
}

.block-container {
  max-width: 1180px;
  padding: 48px 40px !important;
}

h1, h2, h3 {
  font-family: 'Sora', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: -0.03em !important;
  color: #e8e8f0 !important;
}

.section-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.2em;
  color: #4a4a60;
  text-transform: uppercase;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}

.metric-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 20px 24px;
  transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(255,255,255,0.14); }

.metric-value {
  font-size: 28px;
  font-weight: 600;
  color: #e8e8f0;
  letter-spacing: -0.03em;
  margin-bottom: 4px;
}

.metric-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: #4a4a60;
  text-transform: uppercase;
}

.subtle-copy {
  color: #7a7a96;
  font-size: 14px;
  line-height: 1.7;
}

.stTextInput input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 3px !important;
  color: #e8e8f0 !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 300 !important;
  padding: 12px 16px !important;
}

.answer-box {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-left: 2px solid #c8d4f0;
  padding: 20px 24px;
  font-size: 14px;
  font-weight: 300;
  color: #7a7a96;
  line-height: 1.75;
  margin-top: 16px;
}

.legend-panel {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 18px 18px 10px;
  height: 560px;
  overflow-y: auto;
}

.legend-title {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: #7a7a96;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  flex: 0 0 10px;
  border: 1px solid rgba(255,255,255,0.18);
}

.legend-name {
  color: #e8e8f0;
  font-size: 12px;
  line-height: 1.4;
}

.legend-count {
  color: #7a7a96;
  font-size: 11px;
}

.signal-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  margin-bottom: 30px;
}

.signal-legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 10px 12px;
  border-radius: 4px;
}

.signal-legend-item img {
  width: 84px;
  height: 28px;
  display: block;
}

.signal-legend-copy {
  color: #e8e8f0;
  font-size: 12px;
  line-height: 1.35;
}

.detail-block {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 20px 22px;
  height: 100%;
}

.detail-heading {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: #7a7a96;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}

.detail-item:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.detail-key {
  color: #7a7a96;
  font-size: 12px;
  line-height: 1.4;
}

.detail-value {
  color: #e8e8f0;
  font-size: 13px;
  line-height: 1.45;
  text-align: right;
}

.detail-empty {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 26px 28px;
  color: #7a7a96;
}

div[data-baseweb="segmented-control"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 6px !important;
  padding: 4px !important;
  width: 100% !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
}

div[data-baseweb="segmented-control"] > div {
  display: grid !important;
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  gap: 4px !important;
  width: 100% !important;
}

div[data-baseweb="segmented-control"] button {
  color: #7a7a96 !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  width: 100% !important;
  min-width: 0 !important;
  min-height: 46px !important;
  justify-content: center !important;
  border-radius: 4px !important;
  white-space: nowrap !important;
}

div[data-baseweb="segmented-control"] button[aria-checked="true"] {
  background: #7aa6d8 !important;
  color: #0a0a0f !important;
  box-shadow: 0 0 0 1px rgba(122,166,216,0.32) !important;
}

div[data-baseweb="segmented-control"] button:hover {
  background: rgba(255,255,255,0.06) !important;
  color: #e8e8f0 !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 4px;
  overflow: hidden;
}

footer, #MainMenu, header { display: none !important; }
</style>
""", unsafe_allow_html=True)


def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def build_gradient(start_hex, end_hex, steps):
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)

    if steps <= 1:
        return [start_hex]

    colors = []
    for step in range(steps):
        ratio = step / (steps - 1)
        rgb = tuple(
            round(start_channel + (end_channel - start_channel) * ratio)
            for start_channel, end_channel in zip(start_rgb, end_rgb)
        )
        colors.append(rgb_to_hex(rgb))
    return colors


def svg_to_data_uri(svg_markup):
    encoded = base64.b64encode(svg_markup.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def build_signal_badge_svg(is_deepest, is_water_depth, is_dry_rate_flag):
    active = '#7aa6d8'
    inactive = '#5b6178'

    def badge(circle_x, icon, enabled):
        stroke = active if enabled else inactive
        fill_opacity = '0.18' if enabled else '0.10'
        return f"""
        <circle cx="{circle_x}" cy="14" r="10" fill="{stroke}" fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-width="1.3" />
        <text x="{circle_x}" y="14" text-anchor="middle" dominant-baseline="central" font-family="Arial, sans-serif" font-size="11" fill="{stroke}">{icon}</text>
        """

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="84" height="28" viewBox="0 0 84 28">
      {badge(14, '↓', is_deepest)}
      {badge(42, '≈', is_water_depth)}
      {badge(70, '×', is_dry_rate_flag)}
    </svg>
    """
    return svg_to_data_uri(svg)


def render_signal_legend():
    legend_items = [
        (build_signal_badge_svg(True, False, False), 'Top 10 deepest wells'),
        (build_signal_badge_svg(False, True, False), 'Top 10 highest water-depth wells'),
        (build_signal_badge_svg(False, False, True), 'Top 10 dry wells by operator dry-well rate'),
    ]
    legend_cols = st.columns(3, gap='small')
    for col, (icon_src, label) in zip(legend_cols, legend_items):
        with col:
            st.markdown(
                f"""
                <div class="signal-legend-item">
                    <img src="{icon_src}" alt="{html.escape(label)}">
                    <div class="signal-legend-copy">{html.escape(label)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def has_real_value(series):
    as_text = series.astype(str).str.strip()
    invalid_values = {"", "nan", "None", "<NA>"}
    return series.notna() & ~as_text.isin(invalid_values)


def prepare_well_table(df):
    well_table = df.copy()
    fallback_well_ids = pd.Series(well_table.index.astype(str), index=well_table.index)
    well_table['Well ID'] = well_table['NPDID wellbore'].fillna(fallback_well_ids).astype(str)
    well_table['Operator'] = well_table['Drilling operator'].fillna('Unknown')
    well_table['Field / Discovery'] = (
        well_table['Field'].fillna('').astype(str).str.strip()
        .where(has_real_value(well_table['Field']), well_table['Discovery'].fillna('').astype(str).str.strip())
    )
    numeric_columns = [
        'Water depth [m]',
        'Total depth (MD) [m RKB]',
        'Final vertical depth (TVD) [m RKB]',
        'NS decimal degrees',
        'EW decimal degrees',
        'Drilling days',
        'Maximum inclination [°]',
        'Bottom hole temperature [°C]',
    ]
    for column in numeric_columns:
        if column in well_table.columns:
            well_table[column] = pd.to_numeric(well_table[column], errors='coerce')
    well_table['Is Dry'] = well_table['Content'].eq('DRY')
    return well_table


def format_detail_value(value, decimals=0, use_grouping=True):
    if pd.isna(value):
        return '—'
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}" if use_grouping else str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return '—'
        if decimals == 0:
            rounded_value = int(round(value))
            return f"{rounded_value:,}" if use_grouping else str(rounded_value)
        return f"{value:,.{decimals}f}"
    text = str(value).strip()
    return text if text and text not in {"nan", "None", "<NA>"} else '—'


def render_info_cards(items, columns=4):
    cols = st.columns(columns)
    for index, (label, value) in enumerate(items):
        cols[index % columns].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-label">{html.escape(label)}</div>
        </div>""", unsafe_allow_html=True)


def render_detail_block(title, items):
    rows = ''.join(
        f"""
        <div class="detail-item">
            <div class="detail-key">{html.escape(label)}</div>
            <div class="detail-value">{html.escape(str(value))}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f"""
        <div class="detail-block">
            <div class="detail-heading">{html.escape(title)}</div>
            {rows}
        </div>
        """,
        unsafe_allow_html=True
    )


def extract_selected_rows(selection_event):
    if selection_event is None or not hasattr(selection_event, 'selection'):
        return []
    selection = selection_event.selection
    if hasattr(selection, 'rows'):
        return list(selection.rows)
    if isinstance(selection, dict):
        return list(selection.get('rows', []))
    try:
        return list(selection['rows'])
    except Exception:
        return []


def haversine_distance_km(lat1, lon1, lat2_series, lon2_series):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2_series.astype(float))
    lon2_rad = np.radians(lon2_series.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def get_nearby_wells(well_table, selected_well_id, limit=6):
    selected_row = well_table[well_table['Well ID'] == selected_well_id]
    if selected_row.empty:
        return pd.DataFrame()
    selected_well = selected_row.iloc[0]
    if pd.isna(selected_well['NS decimal degrees']) or pd.isna(selected_well['EW decimal degrees']):
        return pd.DataFrame()

    nearby = well_table.dropna(subset=['NS decimal degrees', 'EW decimal degrees']).copy()
    nearby = nearby[nearby['Well ID'] != selected_well_id]
    if nearby.empty:
        return nearby

    nearby['Distance (km)'] = haversine_distance_km(
        selected_well['NS decimal degrees'],
        selected_well['EW decimal degrees'],
        nearby['NS decimal degrees'],
        nearby['EW decimal degrees']
    )
    return nearby.sort_values('Distance (km)').head(limit)


def build_well_detail_map(selected_well, nearby_wells):
    if pd.isna(selected_well['NS decimal degrees']) or pd.isna(selected_well['EW decimal degrees']):
        return None

    selected_lat = float(selected_well['NS decimal degrees'])
    selected_lon = float(selected_well['EW decimal degrees'])
    map_fig = go.Figure()

    if not nearby_wells.empty:
        map_fig.add_trace(
            go.Scattergeo(
                lon=nearby_wells['EW decimal degrees'],
                lat=nearby_wells['NS decimal degrees'],
                mode='markers',
                text=nearby_wells['Wellbore name'],
                customdata=nearby_wells[['Operator', 'Distance (km)']],
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Operator: %{customdata[0]}<br>'
                    'Distance: %{customdata[1]:.1f} km<extra></extra>'
                ),
                marker=dict(size=7, color='rgba(122,166,216,0.35)', line=dict(color='rgba(255,255,255,0.10)', width=0.4)),
                showlegend=False
            )
        )

    map_fig.add_trace(
        go.Scattergeo(
            lon=[selected_lon],
            lat=[selected_lat],
            mode='markers',
            text=[selected_well['Wellbore name']],
            customdata=[[selected_well['Operator'], selected_well['Status'], selected_well['Content']]],
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Operator: %{customdata[0]}<br>'
                'Status: %{customdata[1]}<br>'
                'Content: %{customdata[2]}<extra></extra>'
            ),
            marker=dict(size=13, color='#7aa6d8', line=dict(color='rgba(255,255,255,0.55)', width=1.2)),
            showlegend=False
        )
    )

    if nearby_wells.empty:
        lat_margin = 2.2
        lon_margin = 3.0
    else:
        all_lats = pd.concat([nearby_wells['NS decimal degrees'], pd.Series([selected_lat])])
        all_lons = pd.concat([nearby_wells['EW decimal degrees'], pd.Series([selected_lon])])
        lat_margin = max(1.0, min(4.0, (all_lats.max() - all_lats.min()) * 0.6 + 0.4))
        lon_margin = max(1.4, min(6.0, (all_lons.max() - all_lons.min()) * 0.6 + 0.6))

    map_fig.update_geos(
        projection_type='mercator',
        bgcolor='rgba(0,0,0,0)',
        showland=True,
        landcolor='#141823',
        showocean=True,
        oceancolor='#090c14',
        showcoastlines=True,
        coastlinecolor='rgba(255,255,255,0.18)',
        showcountries=True,
        countrycolor='rgba(255,255,255,0.10)',
        lataxis=dict(range=[max(-90, selected_lat - lat_margin), min(90, selected_lat + lat_margin)]),
        lonaxis=dict(range=[max(-180, selected_lon - lon_margin), min(180, selected_lon + lon_margin)]),
    )
    map_fig.update_layout(
        height=460,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family='Sora', color='#7a7a96', size=11)
    )
    return map_fig


def build_operator_summary(df):
    operator_df = df.copy()
    operator_df['Operator'] = operator_df['Drilling operator'].fillna('Unknown')
    operator_df['Is P&A'] = operator_df['Status'].eq('P&A')
    operator_df['Is Dry'] = operator_df['Content'].eq('DRY')
    operator_df['Has HC'] = has_real_value(operator_df['Content']) & ~operator_df['Content'].eq('DRY')
    operator_df['Has Discovery'] = has_real_value(operator_df['Discovery'])
    operator_df['Water depth [m]'] = pd.to_numeric(operator_df['Water depth [m]'], errors='coerce')
    operator_df['Total depth (MD) [m RKB]'] = pd.to_numeric(operator_df['Total depth (MD) [m RKB]'], errors='coerce')

    summary = (
        operator_df.groupby('Operator', dropna=False)
        .agg(
            **{
                '# Wells': ('Operator', 'size'),
                '# P&A Wells': ('Is P&A', 'sum'),
                '# Dry Wells': ('Is Dry', 'sum'),
                '# HC Wells': ('Has HC', 'sum'),
                '# Discoveries': ('Has Discovery', 'sum'),
                'Avg Water Depth (m)': ('Water depth [m]', 'mean'),
                'Avg Total Depth (m RKB)': ('Total depth (MD) [m RKB]', 'mean'),
                'First Well Year': ('Year', 'min'),
                'Latest Well Year': ('Year', 'max'),
            }
        )
        .reset_index()
        .rename(columns={'Operator': 'Operator'})
    )

    summary['P&A Share (%)'] = (summary['# P&A Wells'] / summary['# Wells'] * 100).round(1)
    summary['Dry Share (%)'] = (summary['# Dry Wells'] / summary['# Wells'] * 100).round(1)
    summary['Discovery Share (%)'] = (summary['# Discoveries'] / summary['# Wells'] * 100).round(1)
    summary = summary.sort_values(['# Wells', '# Discoveries'], ascending=[False, False]).reset_index(drop=True)

    ordered_columns = [
        'Operator',
        '# Wells',
        '# Discoveries',
        'Discovery Share (%)',
        '# HC Wells',
        '# Dry Wells',
        'Dry Share (%)',
        '# P&A Wells',
        'P&A Share (%)',
        'Avg Water Depth (m)',
        'Avg Total Depth (m RKB)',
        'First Well Year',
        'Latest Well Year',
    ]
    return summary[ordered_columns]


def render_page_header(current_page):
    label_col, nav_col = st.columns([2.7, 3.3], vertical_alignment='center')
    with label_col:
        st.markdown(
            '<div class="section-label">Norwegian Continental Shelf · SODIR × Claude</div>',
            unsafe_allow_html=True
        )
    with nav_col:
        selected_page = st.segmented_control(
            "Page",
            options=["Overview", "Operators", "Wells", "Well Detail"],
            key="page",
            label_visibility="collapsed",
            width="stretch",
        )

    active_page = selected_page or current_page

    if active_page == "Overview":
        st.markdown("<h1 style='font-size:40px; margin-bottom:8px;'>Norwegian Shelf Explorer</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#7a7a96; font-size:15px; margin-bottom:48px; font-weight:300;'>AI-powered intelligence over NCS exploration well data</p>",
            unsafe_allow_html=True
        )
    elif active_page == "Operators":
        st.markdown("<h1 style='font-size:40px; margin-bottom:8px;'>Operators</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#7a7a96; font-size:15px; margin-bottom:48px; font-weight:300;'>Sortable SODIR operator table for benchmarking drilling activity across the Norwegian shelf</p>",
            unsafe_allow_html=True
        )
    elif active_page == "Wells":
        st.markdown("<h1 style='font-size:40px; margin-bottom:8px;'>Wells</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#7a7a96; font-size:15px; margin-bottom:48px; font-weight:300;'>Searchable SODIR well inventory with key operational, geological, and location fields</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<h1 style='font-size:40px; margin-bottom:8px;'>Well Detail</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#7a7a96; font-size:15px; margin-bottom:48px; font-weight:300;'>Single-well profile with SODIR metadata, zoomed location, links, and nearby-well context</p>",
            unsafe_allow_html=True
        )
    return active_page


def render_metric_cards(stats):
    col1, col2, col3, col4 = st.columns(4)
    for col, value, label in zip(
        [col1, col2, col3, col4],
        [stats["total_wells"], f"{stats['pa_wells']} ({stats['pa_pct']}%)", stats["dry_wells"], stats["unique_operators"]],
        ["Total Wells", "Plugged & Abandoned", "Dry Wells", "Unique Operators"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)


def render_well_map(df, field_outlines):
    mapped_wells = df.dropna(subset=['NS decimal degrees', 'EW decimal degrees']).copy()
    mapped_wells['Operator'] = mapped_wells['Drilling operator'].fillna('Unknown')

    operator_counts = mapped_wells['Operator'].value_counts().sort_values(ascending=False)
    operator_colors = dict(
        zip(operator_counts.index, build_gradient('#7c1018', '#7aa6d8', len(operator_counts)))
    )
    field_line_color = '#b66d5d'
    field_label_color = 'rgba(214, 180, 173, 0.68)'
    country_labels = [
        {"name": "Norway", "lat": 64.8, "lon": 11.0},
        {"name": "Sweden", "lat": 62.0, "lon": 16.0},
        {"name": "Finland", "lat": 64.3, "lon": 26.0},
        {"name": "Denmark", "lat": 56.2, "lon": 10.0},
        {"name": "United Kingdom", "lat": 55.3, "lon": -2.8},
        {"name": "Netherlands", "lat": 52.3, "lon": 5.3},
        {"name": "Germany", "lat": 51.2, "lon": 10.4},
    ]

    overview_lat_range = [55.6, 63.6]
    overview_lon_range = [-10.2, 14.8]

    map_fig = go.Figure()
    field_labels = field_outlines.get('labels', [])
    field_outline_records = field_outlines.get('outlines', field_outlines.get('labels', []))

    for field in field_outline_records:
        for ring in field.get('rings', []):
            lons = [point[0] for point in ring]
            lats = [point[1] for point in ring]
            map_fig.add_trace(
                go.Scattergeo(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(color=field_line_color, width=0.8),
                    name=field['field_name'],
                    text=field['field_name'],
                    customdata=[[field['main_area'], field['field_id']]] * len(ring),
                    hovertemplate=(
                        '<b>%{text}</b><br>'
                        'Area: %{customdata[0]}<br>'
                        'Field ID: %{customdata[1]}<extra></extra>'
                    ),
                    showlegend=False,
                )
            )

    for field in field_labels:
        if pd.notna(field['label_lat']) and pd.notna(field['label_lon']):
            map_fig.add_trace(
                go.Scattergeo(
                    lon=[field['label_lon']],
                    lat=[field['label_lat']],
                    mode='text',
                    text=[field['field_name']],
                    textfont=dict(family='DM Mono', size=7, color=field_label_color),
                    hoverinfo='skip',
                    showlegend=False,
                )
            )

    for operator, operator_df in mapped_wells.groupby('Operator', sort=False):
        map_fig.add_trace(
            go.Scattergeo(
                lon=operator_df['EW decimal degrees'],
                lat=operator_df['NS decimal degrees'],
                mode='markers',
                name=operator,
                text=operator_df['Wellbore name'],
                customdata=operator_df[['Operator', 'Status', 'Content']],
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Operator: %{customdata[0]}<br>'
                    'Status: %{customdata[1]}<br>'
                    'Content: %{customdata[2]}<br>'
                    'Latitude: %{lat:.3f}<br>'
                    'Longitude: %{lon:.3f}<extra></extra>'
                ),
                marker=dict(
                    size=5,
                    color=operator_colors[operator],
                    opacity=0.82,
                    line=dict(color='rgba(255,255,255,0.16)', width=0.35)
                ),
                showlegend=False
            )
        )

    map_fig.add_trace(
        go.Scattergeo(
            lon=[country['lon'] for country in country_labels],
            lat=[country['lat'] for country in country_labels],
            mode='text',
            text=[country['name'] for country in country_labels],
            textfont=dict(family='DM Mono', size=11, color='rgba(232,232,240,0.42)'),
            hoverinfo='skip',
            showlegend=False
        )
    )
    map_fig.update_geos(
        projection_type='mercator',
        bgcolor='rgba(0,0,0,0)',
        showland=True,
        landcolor='#141823',
        showocean=True,
        oceancolor='#0f1116',
        showlakes=True,
        lakecolor='#0f1116',
        showcoastlines=True,
        coastlinecolor='rgba(255,255,255,0.22)',
        showcountries=True,
        countrycolor='rgba(255,255,255,0.12)',
        showsubunits=True,
        subunitcolor='rgba(255,255,255,0.08)',
        lataxis=dict(range=overview_lat_range),
        lonaxis=dict(range=overview_lon_range)
    )
    map_fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family='Sora', color='#7a7a96', size=11)
    )

    map_col, legend_col = st.columns([5.9, 1.3], gap='small')
    with map_col:
        st.plotly_chart(map_fig, use_container_width=True)

    with legend_col:
        legend_items = ''.join(
            f"""
            <div class="legend-item">
                <span class="legend-swatch" style="background:{operator_colors[operator]};"></span>
                <div>
                    <div class="legend-name">{html.escape(str(operator))}</div>
                    <div class="legend-count">{count} wells</div>
                </div>
            </div>
            """
            for operator, count in operator_counts.items()
        )
        st.markdown(
            f"""
            <div class="legend-panel">
                <div class="legend-title">Operators</div>
                {legend_items}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_oil_price_section():
    st.markdown('<div class="section-label">Global oil price — Daily Brent crude price (USD/barrel)</div>', unsafe_allow_html=True)

    with st.spinner(""):
        oil_df = load_oil_prices()

    if oil_df.empty:
        st.markdown(
            "<div class='detail-empty'>Oil price data is unavailable because `EIA_API_KEY` is not configured for this deployment.</div>",
            unsafe_allow_html=True
        )
        return

    oil_reset = oil_df.reset_index()
    oil_reset.columns = ['Date', 'WTI', 'Brent']
    fig = go.Figure()
    band_count = 12
    band_levels = np.linspace(1 / band_count, 1, band_count)
    base_rgb = (200, 212, 240)

    for idx, level in enumerate(band_levels, start=1):
        opacity = 0.01 + (idx / band_count) * 0.11
        fig.add_trace(
            go.Scatter(
                x=oil_reset['Date'],
                y=oil_reset['Brent'] * level,
                mode='lines',
                line=dict(width=0, color='rgba(0,0,0,0)'),
                fill='tozeroy' if idx == 1 else 'tonexty',
                fillcolor=f'rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {opacity:.3f})',
                hoverinfo='skip',
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=oil_reset['Date'],
            y=oil_reset['Brent'],
            mode='lines',
            name='Brent',
            line=dict(color='#c8d4f0', width=2.4),
            hovertemplate='%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>',
        )
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Sora', color='#7a7a96', size=11),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(255,255,255,0.22)',
            linewidth=1,
            color='#4a4a60',
            title='',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            showline=True,
            linecolor='rgba(255,255,255,0.22)',
            linewidth=1,
            color='#4a4a60',
            title='USD / barrel',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(title_text='', font=dict(color='#7a7a96'), bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)


def render_overview_page(df, stats):
    render_metric_cards(stats)
    st.markdown("<div style='margin: 48px 0 0;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Exploration wells mapped by SODIR coordinates</div>', unsafe_allow_html=True)
    try:
        field_outlines = load_field_outlines()
    except Exception:
        field_outlines = {"geojson": {"type": "FeatureCollection", "features": []}, "labels": [], "outlines": []}
    render_well_map(df, field_outlines)

    render_oil_price_section()

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-label">Wells drilled per year</div>', unsafe_allow_html=True)
        yearly = df.groupby('Year').size().reset_index(name='count')
        fig = px.bar(yearly, x='Year', y='count', color_discrete_sequence=['#c8d4f0'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Sora', color='#7a7a96', size=11),
            xaxis=dict(showgrid=False, color='#4a4a60'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#4a4a60'),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Well content breakdown</div>', unsafe_allow_html=True)
        content = df['Content'].value_counts().reset_index()
        content.columns = ['Content', 'count']
        fig = px.pie(
            content,
            names='Content',
            values='count',
            color_discrete_sequence=['#c8d4f0', '#7a8ab0', '#4a5a80', '#2a3a60', '#1a2a50', '#0a1a40']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Sora', color='#7a7a96', size=11),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(font=dict(color='#7a7a96'))
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Ask Claude about the data</div>', unsafe_allow_html=True)
    question = st.text_input(
        "Question",
        label_visibility="collapsed",
        placeholder="e.g. Which operators have the most P&A wells?"
    )

    if question:
        with st.spinner(""):
            answer = ask_claude(question, df)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)


def render_operators_page(df):
    operator_summary = build_operator_summary(df)
    operator_summary = operator_summary.copy()
    operator_summary['Select'] = operator_summary['Operator'].isin(st.session_state.get('selected_operator_ids', []))

    top_operator = operator_summary.iloc[0]
    card1, card2, card3 = st.columns(3)
    summary_cards = [
        (card1, top_operator['Operator'], 'Most Wells'),
        (card2, int(operator_summary['# Discoveries'].max()), 'Top Discovery Count'),
        (card3, int(operator_summary['Operator'].nunique()), 'Operators Covered'),
    ]
    for col, value, label in summary_cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin: 32px 0 0;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Operator benchmarking table</div>', unsafe_allow_html=True)
    st.markdown(
        "<p class='subtle-copy'>Click any column header to sort the operator ranking. The table starts ordered by total wells so the most active operators appear first.</p>",
        unsafe_allow_html=True
    )

    operator_ids = operator_summary['Operator'].tolist()
    selected_operator_ids = [
        operator for operator in st.session_state.get('selected_operator_ids', [])
        if operator in set(operator_ids)
    ]
    st.session_state['selected_operator_ids'] = selected_operator_ids
    operator_summary['Select'] = operator_summary['Operator'].isin(selected_operator_ids)

    controls_col, download_col, selection_info_col = st.columns([1.1, 1.3, 3.6], vertical_alignment='center')
    with controls_col:
        with st.popover("Select ▾", use_container_width=True):
            if st.button("Select all", use_container_width=True, key="select_all_operators"):
                st.session_state['selected_operator_ids'] = operator_ids
                st.rerun()
            if st.button("Drop all", use_container_width=True, key="drop_all_operators"):
                st.session_state['selected_operator_ids'] = []
                st.rerun()

    selected_download_df = operator_summary[operator_summary['Operator'].isin(selected_operator_ids)].drop(columns=['Select']).copy()
    csv_bytes = selected_download_df.to_csv(index=False).encode('utf-8')
    with download_col:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="selected_operators.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=selected_download_df.empty,
        )
    with selection_info_col:
        st.markdown(
            f"<p class='subtle-copy'>{len(selected_operator_ids)} operator(s) selected.</p>",
            unsafe_allow_html=True
        )

    editor_columns = ['Select'] + [column for column in operator_summary.columns if column != 'Select']
    edited_df = st.data_editor(
        operator_summary[editor_columns],
        use_container_width=True,
        hide_index=True,
        height=680,
        disabled=[column for column in editor_columns if column != 'Select'],
        key="operators_table_editor",
        column_config={
            'Select': st.column_config.CheckboxColumn('Select', width='small'),
            'Operator': st.column_config.TextColumn('Operator', width='medium'),
            '# Wells': st.column_config.NumberColumn('# Wells', format='%d'),
            '# Discoveries': st.column_config.NumberColumn('# Discoveries', format='%d'),
            'Discovery Share (%)': st.column_config.NumberColumn('Discovery Share (%)', format='%.1f'),
            '# HC Wells': st.column_config.NumberColumn('# HC Wells', format='%d'),
            '# Dry Wells': st.column_config.NumberColumn('# Dry Wells', format='%d'),
            'Dry Share (%)': st.column_config.NumberColumn('Dry Share (%)', format='%.1f'),
            '# P&A Wells': st.column_config.NumberColumn('# P&A Wells', format='%d'),
            'P&A Share (%)': st.column_config.NumberColumn('P&A Share (%)', format='%.1f'),
            'Avg Water Depth (m)': st.column_config.NumberColumn('Avg Water Depth (m)', format='%.0f'),
            'Avg Total Depth (m RKB)': st.column_config.NumberColumn('Avg Total Depth (m RKB)', format='%.0f'),
            'First Well Year': st.column_config.NumberColumn('First Well Year', format='%d'),
            'Latest Well Year': st.column_config.NumberColumn('Latest Well Year', format='%d'),
        }
    )

    updated_selected_operator_ids = edited_df.loc[edited_df['Select'].fillna(False), 'Operator'].tolist()
    if updated_selected_operator_ids != st.session_state.get('selected_operator_ids', []):
        st.session_state['selected_operator_ids'] = updated_selected_operator_ids
        st.rerun()


def render_wells_page(df):
    well_table = prepare_well_table(df)

    deepest_well_ids = set(
        well_table.dropna(subset=['Total depth (MD) [m RKB]'])
        .nlargest(10, 'Total depth (MD) [m RKB]')['Well ID']
    )
    highest_water_well_ids = set(
        well_table.dropna(subset=['Water depth [m]'])
        .nlargest(10, 'Water depth [m]')['Well ID']
    )
    operator_dry_rates = (
        well_table.groupby('Operator', dropna=False)
        .agg(
            well_count=('Operator', 'size'),
            dry_wells=('Is Dry', 'sum')
        )
        .query('well_count >= 5')
        .assign(dry_rate=lambda frame: frame['dry_wells'] / frame['well_count'])
        .sort_values(['dry_rate', 'dry_wells', 'well_count'], ascending=[False, False, False])
    )
    dry_rate_lookup = operator_dry_rates['dry_rate']
    dry_well_rank = (
        well_table[well_table['Is Dry']].copy()
        .assign(operator_dry_rate=lambda frame: frame['Operator'].map(dry_rate_lookup).fillna(-1))
        .sort_values(
            ['operator_dry_rate', 'Water depth [m]', 'Total depth (MD) [m RKB]', 'Wellbore name'],
            ascending=[False, False, False, True]
        )
    )
    top_dry_rate_well_ids = set(dry_well_rank.head(10)['Well ID'])

    well_table['Exploration signals'] = well_table.apply(
        lambda row: build_signal_badge_svg(
            row['Well ID'] in deepest_well_ids,
            row['Well ID'] in highest_water_well_ids,
            row['Well ID'] in top_dry_rate_well_ids,
        ),
        axis=1
    )

    search_col, operator_col, status_col, content_col = st.columns([2.3, 1.3, 1.0, 1.0])
    search_term = search_col.text_input(
        "Search wells",
        value="",
        placeholder="Search by well name, operator, field, or discovery",
        label_visibility="collapsed"
    )
    operator_options = ['All'] + sorted(well_table['Operator'].dropna().unique().tolist())
    selected_operator = operator_col.selectbox("Operator", operator_options, label_visibility="collapsed")
    status_options = ['All'] + sorted(well_table['Status'].dropna().astype(str).unique().tolist())
    selected_status = status_col.selectbox("Status", status_options, label_visibility="collapsed")
    content_options = ['All'] + sorted(well_table['Content'].dropna().astype(str).unique().tolist())
    selected_content = content_col.selectbox("Content", content_options, label_visibility="collapsed")

    filtered_wells = well_table.copy()
    if search_term:
        search_mask = (
            filtered_wells['Wellbore name'].fillna('').str.contains(search_term, case=False, na=False)
            | filtered_wells['Operator'].fillna('').str.contains(search_term, case=False, na=False)
            | filtered_wells['Field'].fillna('').str.contains(search_term, case=False, na=False)
            | filtered_wells['Discovery'].fillna('').str.contains(search_term, case=False, na=False)
        )
        filtered_wells = filtered_wells[search_mask]
    if selected_operator != 'All':
        filtered_wells = filtered_wells[filtered_wells['Operator'] == selected_operator]
    if selected_status != 'All':
        filtered_wells = filtered_wells[filtered_wells['Status'].astype(str) == selected_status]
    if selected_content != 'All':
        filtered_wells = filtered_wells[filtered_wells['Content'].astype(str) == selected_content]

    card1, card2, card3 = st.columns(3)
    summary_cards = [
        (card1, len(filtered_wells), 'Visible Wells'),
        (card2, int(filtered_wells['Operator'].nunique()), 'Operators in View'),
        (card3, int(filtered_wells['Year'].max()) if not filtered_wells.empty else '—', 'Latest Year in View'),
    ]
    for col, value, label in summary_cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin: 32px 0 0;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Well inventory</div>', unsafe_allow_html=True)
    st.markdown(
        "<p class='subtle-copy'>Click any column header to sort the table. Use the filters above to narrow the well list by operator, status, content, or a free-text search.</p>",
        unsafe_allow_html=True
    )
    render_signal_legend()
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    visible_columns = [
        'Wellbore name',
        'Exploration signals',
        'Year',
        'Operator',
        'Status',
        'Content',
        'Field / Discovery',
        'Main area',
        'Water depth [m]',
        'Total depth (MD) [m RKB]',
        'Drilling facility',
        'Production licence at wellhead',
        'NS decimal degrees',
        'EW decimal degrees',
    ]
    selection_df = filtered_wells.sort_values(['Year', 'Wellbore name'], ascending=[False, True]).reset_index(drop=True)
    display_df = selection_df[visible_columns].copy()

    visible_well_ids = selection_df['Well ID'].tolist()
    selected_well_ids = [
        well_id for well_id in st.session_state.get('selected_well_ids', [])
        if well_id in set(visible_well_ids)
    ]
    st.session_state['selected_well_ids'] = selected_well_ids

    controls_col, download_col, selection_info_col = st.columns([1.1, 1.3, 3.6], vertical_alignment='center')
    with controls_col:
        with st.popover("Select ▾", use_container_width=True):
            if st.button("Select all", use_container_width=True, key="select_all_wells"):
                st.session_state['selected_well_ids'] = visible_well_ids
                st.rerun()
            if st.button("Drop all", use_container_width=True, key="drop_all_wells"):
                st.session_state['selected_well_ids'] = []
                st.rerun()

    download_columns = [column for column in selection_df.columns if column not in {'Is Dry', 'Exploration signals'}]
    selected_download_df = selection_df[selection_df['Well ID'].isin(selected_well_ids)][download_columns].copy()
    csv_bytes = selected_download_df.to_csv(index=False).encode('utf-8')
    with download_col:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="selected_wells.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=selected_download_df.empty,
        )
    with selection_info_col:
        st.markdown(
            f"<p class='subtle-copy'>{len(selected_well_ids)} well(s) selected in the current view.</p>",
            unsafe_allow_html=True
        )

    editor_df = display_df.copy()
    editor_df.insert(0, 'Select', selection_df['Well ID'].isin(selected_well_ids).values)
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        height=720,
        row_height=44,
        disabled=[column for column in editor_df.columns if column != 'Select'],
        key="wells_inventory_editor",
        column_config={
            'Select': st.column_config.CheckboxColumn('Select', width='small'),
            'Wellbore name': st.column_config.TextColumn('Wellbore name', width='medium'),
            'Exploration signals': st.column_config.ImageColumn('Exploration signals', width='medium'),
            'Year': st.column_config.NumberColumn('Year', format='%d'),
            'Operator': st.column_config.TextColumn('Operator', width='medium'),
            'Status': st.column_config.TextColumn('Status', width='small'),
            'Content': st.column_config.TextColumn('Content', width='small'),
            'Field / Discovery': st.column_config.TextColumn('Field / Discovery', width='medium'),
            'Main area': st.column_config.TextColumn('Main area', width='small'),
            'Water depth [m]': st.column_config.NumberColumn('Water depth [m]', format='%.0f'),
            'Total depth (MD) [m RKB]': st.column_config.NumberColumn('Total depth (MD) [m RKB]', format='%.0f'),
            'Drilling facility': st.column_config.TextColumn('Drilling facility', width='medium'),
            'Production licence at wellhead': st.column_config.TextColumn('Production licence at wellhead', width='small'),
            'NS decimal degrees': st.column_config.NumberColumn('NS decimal degrees', format='%.3f'),
            'EW decimal degrees': st.column_config.NumberColumn('EW decimal degrees', format='%.3f'),
        }
    )

    updated_selected_ids = selection_df.loc[edited_df['Select'].fillna(False).to_numpy(), 'Well ID'].tolist()
    if updated_selected_ids != st.session_state.get('selected_well_ids', []):
        st.session_state['selected_well_ids'] = updated_selected_ids
        st.rerun()

    if len(updated_selected_ids) == 1:
        selected_row = selection_df[selection_df['Well ID'] == updated_selected_ids[0]].iloc[0]
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        action_col, info_col = st.columns([1.2, 3.8], vertical_alignment='center')
        with action_col:
            if st.button("Open Well Detail", type="primary", use_container_width=True):
                st.session_state['selected_well_id'] = selected_row['Well ID']
                st.session_state['page'] = 'Well Detail'
                st.rerun()
        with info_col:
            st.markdown(
                f"<p class='subtle-copy'>Selected well: <span style='color:#e8e8f0;'>{html.escape(str(selected_row['Wellbore name']))}</span> · {html.escape(str(selected_row['Operator']))} · {format_detail_value(selected_row['Year'], use_grouping=False)}</p>",
                unsafe_allow_html=True
            )


def render_well_detail_page(df):
    well_table = prepare_well_table(df)
    sorted_options = well_table.sort_values(['Year', 'Wellbore name'], ascending=[False, True]).reset_index(drop=True)
    option_ids = sorted_options['Well ID'].tolist()
    option_labels = {
        row['Well ID']: f"{row['Wellbore name']} · {row['Operator']} · {format_detail_value(row['Year'], use_grouping=False)}"
        for _, row in sorted_options.iterrows()
    }

    if not option_ids:
        st.markdown(
            "<div class='detail-empty'>No wells are available in the current dataset.</div>",
            unsafe_allow_html=True
        )
        return

    selected_well_id = st.session_state.get('selected_well_id')
    if selected_well_id not in set(option_ids):
        selected_well_id = option_ids[0] if option_ids else None
        st.session_state['selected_well_id'] = selected_well_id

    top_col, picker_col = st.columns([1.1, 3.4], vertical_alignment='center')
    with top_col:
        if st.button("Back to Wells", use_container_width=True):
            st.session_state['page'] = 'Wells'
            st.rerun()
    with picker_col:
        selected_well_id = st.selectbox(
            "Select well",
            options=option_ids,
            index=option_ids.index(selected_well_id) if selected_well_id in option_ids else 0,
            format_func=lambda well_id: option_labels[well_id],
            label_visibility="collapsed",
            key="well_detail_selectbox",
        )
        st.session_state['selected_well_id'] = selected_well_id

    if not selected_well_id:
        st.markdown(
            "<div class='detail-empty'>No well is selected yet. Choose one from the Wells page to open its full profile.</div>",
            unsafe_allow_html=True
        )
        return

    selected_row = well_table[well_table['Well ID'] == selected_well_id]
    if selected_row.empty:
        st.markdown(
            "<div class='detail-empty'>This selected well could not be found in the current SODIR dataset.</div>",
            unsafe_allow_html=True
        )
        return

    selected_well = selected_row.iloc[0]
    nearby_wells = get_nearby_wells(well_table, selected_well_id, limit=6)

    st.markdown('<div class="section-label">Selected well</div>', unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='font-size:34px; margin-bottom:8px;'>{html.escape(str(selected_well['Wellbore name']))}</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p class='subtle-copy' style='margin-bottom:28px;'>{html.escape(str(selected_well['Operator']))} · {html.escape(format_detail_value(selected_well['Status']))} · {html.escape(format_detail_value(selected_well['Content']))} · {html.escape(format_detail_value(selected_well['Main area']))}</p>",
        unsafe_allow_html=True
    )

    render_info_cards([
        ('Year', format_detail_value(selected_well['Year'], use_grouping=False)),
        ('Water Depth (m)', format_detail_value(selected_well['Water depth [m]'])),
        ('Total Depth (m RKB)', format_detail_value(selected_well['Total depth (MD) [m RKB]'])),
        ('Drilling Days', format_detail_value(selected_well['Drilling days'])),
    ])

    st.markdown("<div style='margin: 36px 0 0;'></div>", unsafe_allow_html=True)
    map_col, meta_col = st.columns([3.1, 2.1], gap='medium')
    with map_col:
        st.markdown('<div class="section-label">Location & Nearby Wells</div>', unsafe_allow_html=True)
        detail_map = build_well_detail_map(selected_well, nearby_wells)
        if detail_map is None:
            st.markdown(
                "<div class='detail-empty'>This well does not have usable coordinates in the current dataset.</div>",
                unsafe_allow_html=True
            )
        else:
            st.plotly_chart(detail_map, use_container_width=True)

    with meta_col:
        st.markdown('<div class="section-label">Key Context</div>', unsafe_allow_html=True)
        render_detail_block('Identity & Status', [
            ('Operator', format_detail_value(selected_well['Operator'])),
            ('Status', format_detail_value(selected_well['Status'])),
            ('Content', format_detail_value(selected_well['Content'])),
            ('Main area', format_detail_value(selected_well['Main area'])),
            ('Field / discovery', format_detail_value(selected_well['Field / Discovery'])),
        ])

    st.markdown("<div style='margin: 28px 0 0;'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2, gap='medium')
    with left_col:
        st.markdown('<div class="section-label">Geology & Coordinates</div>', unsafe_allow_html=True)
        render_detail_block('Subsurface', [
            ('Oldest penetrated age', format_detail_value(selected_well['Oldest penetrated age'])),
            ('Oldest penetrated formation', format_detail_value(selected_well['Oldest penetrated formation'])),
            ('1st HC formation', format_detail_value(selected_well['1st level with HC, formation'])),
            ('1st HC age', format_detail_value(selected_well['1st level with HC, age'])),
            ('Bottom hole temp (°C)', format_detail_value(selected_well['Bottom hole temperature [°C]'])),
            ('Maximum inclination (°)', format_detail_value(selected_well['Maximum inclination [°]'])),
        ])
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        render_detail_block('Coordinates', [
            ('Latitude', format_detail_value(selected_well['NS decimal degrees'], decimals=3)),
            ('Longitude', format_detail_value(selected_well['EW decimal degrees'], decimals=3)),
            ('Geodetic datum', format_detail_value(selected_well['Geodetic datum'])),
            ('UTM zone', format_detail_value(selected_well['UTM zone'])),
        ])

    with right_col:
        st.markdown('<div class="section-label">Licence, Facility & Links</div>', unsafe_allow_html=True)
        render_detail_block('Operations', [
            ('Production licence at wellhead', format_detail_value(selected_well['Production licence at wellhead'])),
            ('Drilling target licence', format_detail_value(selected_well['Prod. licence for drilling target'])),
            ('Drilling facility', format_detail_value(selected_well['Drilling facility'])),
            ('Facility type', format_detail_value(selected_well['Drilling facility type'])),
            ('Water depth (m)', format_detail_value(selected_well['Water depth [m]'])),
            ('Final TVD (m RKB)', format_detail_value(selected_well['Final vertical depth (TVD) [m RKB]'])),
        ])
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">External Links</div>', unsafe_allow_html=True)
        link_cols = st.columns(3)
        link_specs = [
            ('FactPage', 'FactPage url'),
            ('Factmaps', 'Factmaps'),
            ('Press release', 'Pressrelease url'),
        ]
        for col, (label, column_name) in zip(link_cols, link_specs):
            url = selected_well.get(column_name)
            if pd.notna(url) and str(url).strip() and str(url).strip() not in {'nan', 'None', '<NA>'}:
                col.link_button(label, str(url), use_container_width=True)
            else:
                col.button(label, disabled=True, use_container_width=True)

    st.markdown("<div style='margin: 28px 0 0;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Nearby Wells</div>', unsafe_allow_html=True)
    if nearby_wells.empty:
        st.markdown(
            "<div class='detail-empty'>No nearby wells with valid coordinates were found for this well.</div>",
            unsafe_allow_html=True
        )
    else:
        nearby_display = nearby_wells[
            ['Wellbore name', 'Operator', 'Distance (km)', 'Status', 'Content', 'Water depth [m]', 'Field / Discovery']
        ].copy()
        st.dataframe(
            nearby_display,
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                'Wellbore name': st.column_config.TextColumn('Wellbore name', width='medium'),
                'Operator': st.column_config.TextColumn('Operator', width='medium'),
                'Distance (km)': st.column_config.NumberColumn('Distance (km)', format='%.1f'),
                'Status': st.column_config.TextColumn('Status', width='small'),
                'Content': st.column_config.TextColumn('Content', width='small'),
                'Water depth [m]': st.column_config.NumberColumn('Water depth [m]', format='%.0f'),
                'Field / Discovery': st.column_config.TextColumn('Field / Discovery', width='medium'),
            }
        )


with st.spinner(""):
    df = load_wellbore_data()
    stats = get_summary_stats(df)

if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview'

current_page = st.session_state['page']
current_page = render_page_header(current_page)

if current_page == 'Operators':
    render_operators_page(df)
elif current_page == 'Wells':
    render_wells_page(df)
elif current_page == 'Well Detail':
    render_well_detail_page(df)
else:
    render_overview_page(df, stats)
