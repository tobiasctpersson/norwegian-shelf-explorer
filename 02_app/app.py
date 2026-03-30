import base64
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
  position: relative !important;
  background:
    radial-gradient(circle at 12% 0%, rgba(182,109,93,0.12), rgba(182,109,93,0) 24%),
    radial-gradient(circle at 88% 100%, rgba(122,166,216,0.08), rgba(122,166,216,0) 28%),
    linear-gradient(180deg, rgba(20,24,35,0.88), rgba(9,11,18,0.86)) !important;
  border: 1px solid rgba(255,255,255,0.11) !important;
  border-radius: 20px !important;
  padding: 8px !important;
  width: 100% !important;
  backdrop-filter: blur(18px) saturate(135%);
  -webkit-backdrop-filter: blur(18px) saturate(135%);
  box-shadow:
    0 22px 54px rgba(0,0,0,0.34),
    0 6px 18px rgba(0,0,0,0.18),
    inset 0 1px 0 rgba(255,255,255,0.06),
    inset 0 -1px 0 rgba(255,255,255,0.02) !important;
  overflow: hidden !important;
}

div[data-baseweb="segmented-control"]::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.00) 32%),
    linear-gradient(90deg, rgba(182,109,93,0.00), rgba(182,109,93,0.06), rgba(182,109,93,0.00));
  opacity: 0.9;
}

div[data-baseweb="segmented-control"]::after {
  content: "";
  position: absolute;
  left: 20px;
  right: 20px;
  bottom: -14px;
  height: 28px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(182,109,93,0.22), rgba(182,109,93,0.00) 72%);
  filter: blur(16px);
  pointer-events: none;
}

div[data-baseweb="segmented-control"] > div {
  display: grid !important;
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  gap: 6px !important;
  width: 100% !important;
  position: relative !important;
  z-index: 1 !important;
}

div[data-baseweb="segmented-control"] button {
  color: rgba(232,232,240,0.62) !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
  text-transform: uppercase !important;
  width: 100% !important;
  min-width: 0 !important;
  min-height: 54px !important;
  justify-content: center !important;
  border-radius: 14px !important;
  white-space: nowrap !important;
  border: 1px solid rgba(255,255,255,0.02) !important;
  background: rgba(255,255,255,0.01) !important;
  transition:
    background 0.18s ease,
    color 0.18s ease,
    border-color 0.18s ease,
    transform 0.18s ease,
    box-shadow 0.18s ease !important;
}

div[data-baseweb="segmented-control"] button[aria-checked="true"] {
  background:
    linear-gradient(120deg, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0.02) 18%, rgba(255,255,255,0.00) 42%),
    radial-gradient(circle at 50% 0%, rgba(255,255,255,0.16), rgba(255,255,255,0) 56%),
    linear-gradient(180deg, rgba(198,122,103,0.98), rgba(156,87,72,0.96)) !important;
  color: #fff6f2 !important;
  border-color: rgba(233,178,162,0.36) !important;
  box-shadow:
    0 16px 30px rgba(131,64,49,0.30),
    inset 0 1px 0 rgba(255,255,255,0.18),
    0 0 0 1px rgba(182,109,93,0.16),
    0 0 24px rgba(182,109,93,0.18) !important;
  transform: translateY(-1px) scale(1.01);
}

div[data-baseweb="segmented-control"] button:hover {
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03)) !important;
  color: #e8e8f0 !important;
  border-color: rgba(255,255,255,0.08) !important;
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(0,0,0,0.18) !important;
}

div[data-baseweb="segmented-control"] button[aria-checked="true"]:hover {
  color: #fff6f2 !important;
}

div[data-baseweb="segmented-control"] button p {
  margin: 0 !important;
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


@st.cache_data(show_spinner=False)
def get_video_data_uri(video_path):
    video_bytes = Path(video_path).read_bytes()
    encoded = base64.b64encode(video_bytes).decode('utf-8')
    return f"data:video/mp4;base64,{encoded}"


def render_overview_hero():
    video_path = Path(__file__).resolve().parent.parent / "oil_rig_video.mp4"
    video_data_uri = get_video_data_uri(str(video_path))
    hero_html = f"""
    <html>
      <head>
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
          }}

          .hero-shell {{
            position: relative;
            height: 76vh;
            min-height: 540px;
            max-height: 760px;
            border: 1px solid rgba(255,255,255,0.08);
            overflow: hidden;
            background: #05070b;
          }}

          .hero-media-wrap {{
            position: absolute;
            inset: 0;
            transition: opacity 0.12s linear, transform 0.18s ease-out;
            will-change: opacity, transform;
          }}

          .hero-video {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: saturate(0.78) brightness(0.62) contrast(1.04);
          }}

          .hero-scrim {{
            position: absolute;
            inset: 0;
            background:
              radial-gradient(circle at 50% 48%, rgba(5,7,11,0.18), rgba(5,7,11,0.58) 58%, rgba(5,7,11,0.88) 100%),
              linear-gradient(180deg, rgba(5,7,11,0.22), rgba(5,7,11,0.48) 65%, rgba(5,7,11,0.82));
          }}

          .hero-copy {{
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 32px;
          }}

          .hero-kicker {{
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: rgba(200,212,240,0.68);
            margin-bottom: 20px;
          }}

          .hero-title {{
            font-family: 'Sora', sans-serif;
            font-size: clamp(46px, 7vw, 86px);
            line-height: 0.95;
            letter-spacing: -0.05em;
            color: #f3f5fb;
            margin: 0;
            text-shadow: 0 12px 40px rgba(0,0,0,0.35);
          }}

          .hero-subtitle {{
            margin-top: 18px;
            max-width: 760px;
            font-family: 'Sora', sans-serif;
            font-size: 14px;
            line-height: 1.7;
            color: rgba(232,232,240,0.72);
          }}
        </style>
      </head>
      <body>
        <section class="hero-shell">
          <div class="hero-media-wrap" id="heroMedia">
            <video class="hero-video" id="heroVideo" autoplay muted loop playsinline preload="auto">
              <source src="{video_data_uri}" type="video/mp4" />
            </video>
            <div class="hero-scrim"></div>
          </div>
          <div class="hero-copy">
            <div class="hero-kicker">Norwegian Continental Shelf Intelligence</div>
            <h1 class="hero-title">Norwegian Shelf Explorer</h1>
            <div class="hero-subtitle">Exploration wells, operator activity, and market context across the NCS.</div>
          </div>
        </section>
        <script>
          const video = document.getElementById("heroVideo");
          const media = document.getElementById("heroMedia");
          if (video) {{
            video.playbackRate = 0.55;
          }}

          const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

          function updateHeroOpacity() {{
            const frame = window.frameElement;
            if (!frame || !frame.getBoundingClientRect) return;

            const rect = frame.getBoundingClientRect();
            const parentWindow = window.parent || window;
            const viewportHeight = parentWindow.innerHeight || window.innerHeight || 1;
            const progress = clamp((-rect.top) / (viewportHeight * 0.78), 0, 1);
            const opacity = clamp(1 - progress * 1.12, 0, 1);
            const scale = 1 + progress * 0.035;

            media.style.opacity = opacity.toFixed(3);
            media.style.transform = `scale(${{scale.toFixed(3)}})`;
          }}

          try {{
            window.parent.addEventListener("scroll", updateHeroOpacity, {{ passive: true }});
          }} catch (error) {{
            window.addEventListener("scroll", updateHeroOpacity, {{ passive: true }});
          }}
          window.addEventListener("resize", updateHeroOpacity);
          updateHeroOpacity();
          setInterval(updateHeroOpacity, 220);
        </script>
      </body>
    </html>
    """
    components.html(hero_html, height=640, scrolling=False)


def _build_svg_path(points):
    if not points:
        return ""
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)


def render_scroll_reveal_brent_chart(oil_df):
    chart_df = oil_df[['Brent']].dropna().reset_index()
    chart_df.columns = ['Date', 'Brent']
    chart_df['Date'] = pd.to_datetime(chart_df['Date'])
    chart_df['Brent'] = pd.to_numeric(chart_df['Brent'], errors='coerce')
    chart_df = chart_df.dropna(subset=['Date', 'Brent']).reset_index(drop=True)

    width = 1180
    height = 560
    margin_left = 92
    margin_right = 34
    margin_top = 28
    margin_bottom = 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_height

    latest_row = chart_df.iloc[-1]
    peak_row = chart_df.loc[chart_df['Brent'].idxmax()]
    median_price = chart_df['Brent'].median()
    trailing_window = chart_df.tail(min(252, len(chart_df)))
    trailing_avg = trailing_window['Brent'].mean()
    first_row = chart_df.iloc[0]

    y_max = max(160, int(math.ceil(chart_df['Brent'].max() / 10.0) * 10))
    y_min = 0

    date_numbers = chart_df['Date'].astype('int64').to_numpy()
    if len(date_numbers) > 1:
        x_positions = np.interp(date_numbers, (date_numbers.min(), date_numbers.max()), (margin_left, margin_left + plot_width))
    else:
        x_positions = np.array([margin_left + plot_width / 2], dtype=float)

    y_positions = np.interp(chart_df['Brent'].to_numpy(), (y_min, y_max), (baseline_y, margin_top))
    points = list(zip(x_positions, y_positions))
    line_path = _build_svg_path(points)
    area_path = (
        f"{line_path} "
        f"L {x_positions[-1]:.2f} {baseline_y:.2f} "
        f"L {x_positions[0]:.2f} {baseline_y:.2f} Z"
    )

    y_ticks = list(range(0, y_max + 1, 20))
    if y_ticks[-1] != y_max:
        y_ticks.append(y_max)

    start_year = int(chart_df['Date'].dt.year.min())
    end_year = int(chart_df['Date'].dt.year.max())
    tick_years = list(range(1990, 2026, 5))

    start_date = chart_df['Date'].iloc[0]
    end_date = chart_df['Date'].iloc[-1]
    span_ns = max(end_date.value - start_date.value, 1)

    x_tick_items = []
    for year in tick_years:
        tick_date = pd.Timestamp(year=year, month=1, day=1)
        ratio = min(max((tick_date.value - start_date.value) / span_ns, 0), 1)
        x_value = margin_left + ratio * plot_width
        x_tick_items.append((year, x_value))

    payload = {
        "xPositions": [round(value, 2) for value in x_positions.tolist()],
        "yPositions": [round(value, 2) for value in y_positions.tolist()],
        "dates": chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "prices": [round(value, 2) for value in chart_df['Brent'].tolist()],
        "plotLeft": margin_left,
        "plotTop": margin_top,
        "plotWidth": plot_width,
        "plotHeight": plot_height,
    }

    peak_x = x_positions[int(peak_row.name)]
    peak_y = y_positions[int(peak_row.name)]
    latest_x = x_positions[-1]
    latest_y = y_positions[-1]
    latest_date_label = pd.Timestamp(latest_row['Date']).strftime('%Y-%m-%d')
    peak_date_label = pd.Timestamp(peak_row['Date']).strftime('%Y-%m-%d')
    stats_markup = f"""
      <div class="chart-topline">
        <div class="market-stat">
          <span class="market-label">Latest</span>
          <span class="market-value">${latest_row['Brent']:.2f}</span>
          <span class="market-meta">{latest_date_label}</span>
        </div>
        <div class="market-stat market-stat-accent">
          <span class="market-label">Historical peak</span>
          <span class="market-value">${peak_row['Brent']:.2f}</span>
          <span class="market-meta">{peak_date_label}</span>
        </div>
        <div class="market-stat">
          <span class="market-label">Median since {pd.Timestamp(first_row['Date']).year}</span>
          <span class="market-value">${median_price:.2f}</span>
          <span class="market-meta">Long-run center</span>
        </div>
        <div class="market-stat">
          <span class="market-label">Trailing 252d avg</span>
          <span class="market-value">${trailing_avg:.2f}</span>
          <span class="market-meta">Recent regime</span>
        </div>
      </div>
    """

    y_grid_markup = "\n".join(
        f"""
        <line x1="{margin_left}" y1="{np.interp(tick, (y_min, y_max), (baseline_y, margin_top)):.2f}" x2="{margin_left + plot_width}" y2="{np.interp(tick, (y_min, y_max), (baseline_y, margin_top)):.2f}" class="grid-line" />
        <text x="{margin_left - 14}" y="{np.interp(tick, (y_min, y_max), (baseline_y, margin_top)) + 5:.2f}" class="tick-label tick-left">{tick}</text>
        """
        for tick in y_ticks
    )
    x_tick_markup = "\n".join(
        f"""
        <text x="{x_value:.2f}" y="{baseline_y + 28:.2f}" class="tick-label tick-center">{year}</text>
        """
        for year, x_value in x_tick_items
    )

    chart_html = f"""
    <html>
      <head>
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
            font-family: 'Sora', sans-serif;
          }}

          .chart-shell {{
            position: relative;
            width: 100%;
            height: 620px;
            padding: 18px 18px 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background:
              radial-gradient(circle at 18% 0%, rgba(200,212,240,0.06), rgba(200,212,240,0) 28%),
              radial-gradient(circle at 92% 12%, rgba(182,109,93,0.08), rgba(182,109,93,0) 26%),
              linear-gradient(180deg, rgba(16,18,26,0.94), rgba(8,10,15,0.98));
            box-shadow:
              inset 0 1px 0 rgba(255,255,255,0.03),
              0 28px 50px rgba(0,0,0,0.20);
            overflow: hidden;
          }}

          .chart-shell::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
              linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0) 18%),
              linear-gradient(90deg, rgba(255,255,255,0.018) 0, rgba(255,255,255,0) 22%, rgba(255,255,255,0) 78%, rgba(255,255,255,0.014) 100%);
            pointer-events: none;
          }}

          .chart-topline {{
            position: relative;
            z-index: 2;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
          }}

          .market-stat {{
            display: flex;
            flex-direction: column;
            gap: 5px;
            min-height: 84px;
            padding: 14px 14px 12px;
            border: 1px solid rgba(255,255,255,0.07);
            background: rgba(255,255,255,0.025);
            backdrop-filter: blur(10px);
          }}

          .market-stat-accent {{
            background: linear-gradient(180deg, rgba(182,109,93,0.10), rgba(182,109,93,0.04));
            border-color: rgba(182,109,93,0.20);
          }}

          .market-label {{
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #8f95ac;
          }}

          .market-value {{
            color: #f3f6ff;
            font-size: 24px;
            line-height: 1.05;
            font-weight: 600;
            letter-spacing: -0.03em;
          }}

          .market-meta {{
            color: rgba(232,232,240,0.62);
            font-size: 12px;
            line-height: 1.35;
          }}

          .chart-svg {{
            width: 100%;
            height: calc(100% - 104px);
            display: block;
            position: relative;
            z-index: 1;
          }}

          .grid-line {{
            stroke: rgba(255,255,255,0.045);
            stroke-width: 1;
          }}

          .axis-line {{
            stroke: rgba(255,255,255,0.16);
            stroke-width: 1;
          }}

          .tick-label {{
            fill: #d7dde9;
            font-size: 12px;
            font-family: 'Sora', sans-serif;
            opacity: 0.88;
          }}

          .tick-left {{
            text-anchor: end;
          }}

          .tick-center {{
            text-anchor: middle;
          }}


          .y-label {{
            fill: #eef2fb;
            font-size: 17px;
            font-weight: 600;
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.02em;
          }}

          .legend-label {{
            fill: #a9b1c7;
            font-size: 14px;
            font-weight: 500;
          }}

          .focus-dot {{
            fill: #f3f6ff;
            stroke: rgba(15,17,22,0.72);
            stroke-width: 2;
            display: none;
          }}

          .focus-line {{
            stroke: rgba(255,255,255,0.12);
            stroke-width: 1;
            stroke-dasharray: 4 4;
            display: none;
          }}

          .tooltip {{
            position: absolute;
            pointer-events: none;
            transform: translate(12px, -50%);
            background: rgba(15,17,22,0.94);
            color: #f1f4fb;
            border: 1px solid rgba(255,255,255,0.12);
            padding: 10px 12px;
            font-size: 12px;
            line-height: 1.5;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.12s ease;
            box-shadow: 0 14px 32px rgba(0,0,0,0.24);
          }}

          .terminal-pill {{
            fill: rgba(15,17,22,0.92);
            stroke: rgba(200,212,240,0.16);
            stroke-width: 1;
          }}

          .terminal-pill-accent {{
            fill: rgba(182,109,93,0.12);
            stroke: rgba(182,109,93,0.30);
            stroke-width: 1;
          }}

          .terminal-text {{
            fill: #eef2fb;
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.05em;
          }}

          .annotation-ring {{
            fill: rgba(182,109,93,0.18);
            stroke: rgba(182,109,93,0.72);
            stroke-width: 1.4;
          }}
        </style>
      </head>
      <body>
        <div class="chart-shell" id="chartShell">
          {stats_markup}
          <svg class="chart-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
            <defs>
              <linearGradient id="brentAreaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#d6def4" stop-opacity="0.32" />
                <stop offset="42%" stop-color="#c8d4f0" stop-opacity="0.14" />
                <stop offset="100%" stop-color="#c8d4f0" stop-opacity="0.01" />
              </linearGradient>
              <linearGradient id="lineGlowGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="rgba(200,212,240,0.28)" />
                <stop offset="100%" stop-color="rgba(200,212,240,0.02)" />
              </linearGradient>
              <clipPath id="chartReveal">
                <rect id="revealRect" x="{margin_left}" y="{margin_top}" width="0" height="{plot_height}" />
              </clipPath>
            </defs>

            {y_grid_markup}
            {x_tick_markup}

            <line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + plot_width}" y2="{baseline_y}" class="axis-line" />
            <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline_y}" class="axis-line" />
            <text x="26" y="{margin_top + plot_height / 2:.2f}" class="y-label" transform="rotate(-90 26 {margin_top + plot_height / 2:.2f})">USD / barrel</text>

            <g clip-path="url(#chartReveal)">
              <path d="{area_path}" fill="url(#brentAreaGradient)" opacity="1"></path>
              <path d="{line_path}" fill="none" stroke="rgba(200,212,240,0.14)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"></path>
              <path d="{line_path}" fill="none" stroke="#d8e2f7" stroke-width="3.1" stroke-linecap="round" stroke-linejoin="round"></path>
              <circle cx="{peak_x:.2f}" cy="{peak_y:.2f}" r="6.4" class="annotation-ring"></circle>
              <rect x="{max(margin_left + 12, peak_x - 44):.2f}" y="{max(margin_top + 8, peak_y - 52):.2f}" width="88" height="24" rx="12" class="terminal-pill-accent"></rect>
              <text x="{max(margin_left + 56, peak_x):.2f}" y="{max(margin_top + 23, peak_y - 36):.2f}" text-anchor="middle" class="terminal-text">Peak ${peak_row['Brent']:.0f}</text>
            </g>

            <line id="focusLine" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline_y}" class="focus-line" />
            <circle id="focusDot" cx="{margin_left}" cy="{baseline_y}" r="5.5" class="focus-dot" />

            <g transform="translate({width - 148}, 22)">
              <line x1="0" y1="11" x2="44" y2="11" stroke="#c8d4f0" stroke-width="3" />
              <text x="54" y="16" class="legend-label">Brent</text>
            </g>

            <rect x="{width - 166}" y="{baseline_y - 38:.2f}" width="138" height="24" rx="12" class="terminal-pill"></rect>
            <text x="{width - 97}" y="{baseline_y - 22:.2f}" text-anchor="middle" class="terminal-text">Latest ${latest_row['Brent']:.1f}</text>

            <rect id="hoverOverlay" x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="transparent"></rect>
          </svg>
          <div id="chartTooltip" class="tooltip"></div>
        </div>
        <script>
          const payload = {json.dumps(payload)};
          const revealRect = document.getElementById("revealRect");
          const hoverOverlay = document.getElementById("hoverOverlay");
          const tooltip = document.getElementById("chartTooltip");
          const chartShell = document.getElementById("chartShell");
          const focusDot = document.getElementById("focusDot");
          const focusLine = document.getElementById("focusLine");
          let maxProgress = 0;

          function clamp(value, min, max) {{
            return Math.min(max, Math.max(min, value));
          }}

          function easeOutQuart(value) {{
            return 1 - Math.pow(1 - value, 4);
          }}

          function updateReveal() {{
            const frame = window.frameElement;
            const parentWindow = window.parent || window;
            const viewportHeight = parentWindow.innerHeight || window.innerHeight || 1;
            const rect = frame ? frame.getBoundingClientRect() : chartShell.getBoundingClientRect();
            const rawProgress = (viewportHeight - rect.top - viewportHeight * 0.04) / (viewportHeight * 1.32);
            maxProgress = Math.max(maxProgress, clamp(rawProgress, 0, 1));
            const eased = easeOutQuart(maxProgress);
            revealRect.setAttribute("width", (payload.plotWidth * eased).toFixed(2));
          }}

          function nearestIndex(targetX) {{
            const values = payload.xPositions;
            let low = 0;
            let high = values.length - 1;
            while (low < high) {{
              const mid = Math.floor((low + high) / 2);
              if (values[mid] < targetX) low = mid + 1;
              else high = mid;
            }}
            const current = low;
            const previous = Math.max(0, current - 1);
            return Math.abs(values[current] - targetX) < Math.abs(values[previous] - targetX) ? current : previous;
          }}

          function hideHover() {{
            tooltip.style.opacity = "0";
            focusDot.style.display = "none";
            focusLine.style.display = "none";
          }}

          hoverOverlay.addEventListener("mousemove", (event) => {{
            const bounds = hoverOverlay.getBoundingClientRect();
            const relativeX = event.clientX - bounds.left;
            const scaledX = payload.plotLeft + (relativeX / bounds.width) * payload.plotWidth;
            const revealedWidth = parseFloat(revealRect.getAttribute("width") || "0");
            if (scaledX - payload.plotLeft > revealedWidth) {{
              hideHover();
              return;
            }}

            const index = nearestIndex(scaledX);
            const pointX = payload.xPositions[index];
            const pointY = payload.yPositions[index];
            const localX = ((pointX - payload.plotLeft) / payload.plotWidth) * bounds.width;
            const localY = ((pointY - payload.plotTop) / payload.plotHeight) * bounds.height;

            focusDot.style.display = "block";
            focusLine.style.display = "block";
            focusDot.setAttribute("cx", pointX.toFixed(2));
            focusDot.setAttribute("cy", pointY.toFixed(2));
            focusLine.setAttribute("x1", pointX.toFixed(2));
            focusLine.setAttribute("x2", pointX.toFixed(2));

            tooltip.innerHTML = `${{payload.dates[index]}}<br/>Price: ${{payload.prices[index].toFixed(2)}}`;
            tooltip.style.left = `${{localX}}px`;
            tooltip.style.top = `${{localY}}px`;
            tooltip.style.opacity = "1";
          }});

          hoverOverlay.addEventListener("mouseleave", hideHover);

          try {{
            window.parent.addEventListener("scroll", updateReveal, {{ passive: true }});
          }} catch (error) {{
            window.addEventListener("scroll", updateReveal, {{ passive: true }});
          }}
          window.addEventListener("resize", updateReveal);
          updateReveal();
          setInterval(updateReveal, 180);
        </script>
      </body>
    </html>
    """
    components.html(chart_html, height=640, scrolling=False)


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
        st.markdown("<div style='height: 8px; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
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


def request_page_change(page_name):
    st.session_state['pending_page'] = page_name
    st.rerun()


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


def build_overview_map_html(mapped_wells, field_outline_records, operator_colors):
    field_polygons = []
    for field in field_outline_records:
        latlng_paths = []
        for ring in field.get('rings', []):
            latlng_ring = []
            for point in ring:
                if len(point) < 2:
                    continue
                lon, lat = point[0], point[1]
                if pd.isna(lat) or pd.isna(lon):
                    continue
                latlng_ring.append([float(lat), float(lon)])
            if len(latlng_ring) >= 3:
                latlng_paths.append(latlng_ring)

        if not latlng_paths:
            continue

        field_polygons.append(
            {
                "name": field.get('field_name', 'Unnamed field'),
                "field_id": str(field.get('field_id', '')),
                "main_area": field.get('main_area', 'Unknown'),
                "status": field.get('status', 'Unknown'),
                "label_lat": float(field['label_lat']) if pd.notna(field.get('label_lat')) else None,
                "label_lon": float(field['label_lon']) if pd.notna(field.get('label_lon')) else None,
                "paths": latlng_paths,
            }
        )

    well_records = []
    for _, row in mapped_wells.iterrows():
        lat = row['NS decimal degrees']
        lon = row['EW decimal degrees']
        if pd.isna(lat) or pd.isna(lon):
            continue

        operator = row['Operator']
        well_records.append(
            {
                "name": str(row.get('Wellbore name', 'Unknown well')),
                "operator": str(operator),
                "status": str(row.get('Status', 'Unknown')),
                "content": str(row.get('Content', 'Unknown')),
                "lat": float(lat),
                "lon": float(lon),
                "color": operator_colors.get(operator, '#7aa6d8'),
            }
        )

    lat_padding = 1.15
    lon_padding = 2.2
    bounds = [
        [float(mapped_wells['NS decimal degrees'].min() - lat_padding), float(mapped_wells['EW decimal degrees'].min() - lon_padding)],
        [float(mapped_wells['NS decimal degrees'].max() + lat_padding), float(mapped_wells['EW decimal degrees'].max() + lon_padding)],
    ]

    payload = {
        "fields": field_polygons,
        "wells": well_records,
        "bounds": bounds,
    }

    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin=""
        />
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
            font-family: 'Sora', sans-serif;
          }}

          #overview-map {{
            width: 100%;
            height: 560px;
            border-radius: 24px;
            overflow: hidden;
            background: #0f1116;
            border: 1px solid rgba(255,255,255,0.08);
          }}

          .leaflet-container {{
            background: #0f1116;
            font-family: 'Sora', sans-serif;
          }}

          .leaflet-control-zoom {{
            border: 0 !important;
            box-shadow: 0 18px 34px rgba(0,0,0,0.28) !important;
          }}

          .leaflet-control-zoom a {{
            width: 34px !important;
            height: 34px !important;
            line-height: 34px !important;
            background: rgba(13,15,22,0.94) !important;
            color: #eef2fb !important;
            border-bottom: 1px solid rgba(255,255,255,0.07) !important;
          }}

          .leaflet-control-scale-line {{
            border-color: rgba(255,255,255,0.18) !important;
            color: #d7dde9 !important;
            background: rgba(10,10,15,0.74) !important;
          }}

          .map-hint {{
            position: absolute;
            top: 16px;
            left: 16px;
            z-index: 700;
            padding: 9px 12px;
            border-radius: 999px;
            background: rgba(13,15,22,0.82);
            border: 1px solid rgba(255,255,255,0.08);
            color: rgba(232,232,240,0.78);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            backdrop-filter: blur(8px);
            transition: opacity 0.25s ease;
          }}

          .field-label-marker {{
            background: transparent;
            border: 0;
            opacity: 0;
            transform: translateY(6px);
            transition: opacity 0.22s ease, transform 0.22s ease;
            pointer-events: none;
            white-space: nowrap;
          }}

          .field-label-marker.is-visible {{
            opacity: 1;
            transform: translateY(0);
          }}

          .field-label-text {{
            color: rgba(228, 208, 202, 0.88);
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.08em;
            text-shadow: 0 2px 8px rgba(0,0,0,0.46);
          }}

          .field-tooltip-shell,
          .well-tooltip-shell {{
            background: rgba(13,15,22,0.96);
            color: #eef2fb;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            box-shadow: 0 14px 32px rgba(0,0,0,0.28);
          }}

          .field-tooltip-shell .leaflet-tooltip-content,
          .well-tooltip-shell .leaflet-tooltip-content {{
            margin: 10px 12px;
            font-size: 12px;
            line-height: 1.45;
          }}

          .field-tooltip-shell strong,
          .well-tooltip-shell strong {{
            display: block;
            margin-bottom: 3px;
            font-weight: 600;
          }}
        </style>
      </head>
      <body>
        <div id="overview-map"></div>
        <script
          src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""
        ></script>
        <script>
          const payload = {json.dumps(payload)};

          function escapeHtml(value) {{
            return String(value)
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
          }}

          const map = L.map("overview-map", {{
            zoomControl: true,
            scrollWheelZoom: true,
            wheelPxPerZoomLevel: 18,
            zoomSnap: 0.25,
            zoomDelta: 0.5,
            preferCanvas: true,
            attributionControl: true
          }});

          map.createPane("fieldPane");
          map.getPane("fieldPane").style.zIndex = 410;
          map.createPane("wellPane");
          map.getPane("wellPane").style.zIndex = 520;
          map.createPane("labelPane");
          map.getPane("labelPane").style.zIndex = 560;

          L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
            subdomains: "abcd",
            maxZoom: 18,
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
          }}).addTo(map);

          payload.fields.forEach((field) => {{
            const polygon = L.polygon(field.paths, {{
              pane: "fieldPane",
              color: "#b66d5d",
              weight: 1.15,
              opacity: 0.92,
              fillColor: "#1e1315",
              fillOpacity: 0.34
            }}).addTo(map);

            polygon.bindTooltip(
              `<strong>${{escapeHtml(field.name)}}</strong>Area: ${{escapeHtml(field.main_area)}}<br>Status: ${{escapeHtml(field.status)}}<br>Field ID: ${{escapeHtml(field.field_id)}}`,
              {{
                sticky: true,
                direction: "top",
                className: "field-tooltip-shell",
                opacity: 0.96
              }}
            );

            if (field.label_lat !== null && field.label_lon !== null) {{
              L.marker([field.label_lat, field.label_lon], {{
                pane: "labelPane",
                interactive: false,
                icon: L.divIcon({{
                  className: "field-label-marker",
                  html: `<span class="field-label-text">${{escapeHtml(field.name)}}</span>`,
                  iconSize: [0, 0],
                  iconAnchor: [0, 0]
                }})
              }}).addTo(map);
            }}
          }});

          payload.wells.forEach((well) => {{
            const marker = L.circleMarker([well.lat, well.lon], {{
              pane: "wellPane",
              radius: 4.6,
              color: "rgba(255,255,255,0.16)",
              weight: 1,
              fillColor: well.color,
              fillOpacity: 0.86
            }}).addTo(map);

            marker.bindTooltip(
              `<strong>${{escapeHtml(well.name)}}</strong>Operator: ${{escapeHtml(well.operator)}}<br>Status: ${{escapeHtml(well.status)}}<br>Content: ${{escapeHtml(well.content)}}<br>Latitude: ${{well.lat.toFixed(3)}}<br>Longitude: ${{well.lon.toFixed(3)}}`,
              {{
                sticky: true,
                direction: "top",
                className: "well-tooltip-shell",
                opacity: 0.96
              }}
            );
          }});

          if (Array.isArray(payload.bounds) && payload.bounds.length === 2) {{
            map.fitBounds(payload.bounds, {{ padding: [26, 26] }});
            if (map.getZoom() > 6.25) {{
              map.setZoom(6.25);
            }}
          }} else {{
            map.setView([59.6, 3.0], 5.4);
          }}

          L.control.scale({{ position: "bottomleft", imperial: false }}).addTo(map);

          function updateFieldLabels() {{
            const showLabels = map.getZoom() >= 7.7;
            document.querySelectorAll(".field-label-marker").forEach((element) => {{
              if (showLabels) element.classList.add("is-visible");
              else element.classList.remove("is-visible");
            }});
          }}

          map.on("zoomend", updateFieldLabels);
          map.on("load", updateFieldLabels);
          window.setTimeout(updateFieldLabels, 120);
        </script>
      </body>
    </html>
    """


def render_well_map(df, field_outlines):
    mapped_wells = df.dropna(subset=['NS decimal degrees', 'EW decimal degrees']).copy()
    mapped_wells['Operator'] = mapped_wells['Drilling operator'].fillna('Unknown')

    operator_counts = mapped_wells['Operator'].value_counts().sort_values(ascending=False)
    operator_colors = dict(
        zip(operator_counts.index, build_gradient('#7c1018', '#7aa6d8', len(operator_counts)))
    )
    field_outline_records = field_outlines.get('outlines', field_outlines.get('labels', []))
    map_html = build_overview_map_html(mapped_wells, field_outline_records, operator_colors)

    map_col, legend_col = st.columns([5.9, 1.3], gap='small')
    with map_col:
        components.html(map_html, height=560, scrolling=False)

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
                <div class="legend-title">Oil fields</div>
                <div class="legend-item">
                    <span class="legend-swatch" style="background:#1e1315; border:1px solid #b66d5d; border-radius:3px;"></span>
                    <div>
                        <div class="legend-name">Field outline & area</div>
                        <div class="legend-count">SODIR field polygons</div>
                    </div>
                </div>
                <div style="height: 10px;"></div>
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

    render_scroll_reveal_brent_chart(oil_df)


def build_wells_drilled_chart(df):
    yearly = df.groupby('Year').size().reset_index(name='count').sort_values('Year')
    yearly['rolling_avg'] = yearly['count'].rolling(window=5, min_periods=1).mean()
    peak_row = yearly.loc[yearly['count'].idxmax()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly['Year'],
            y=yearly['count'],
            name='Wells drilled',
            marker=dict(
                color='rgba(200,212,240,0.84)',
                line=dict(color='rgba(255,255,255,0.10)', width=0.6),
            ),
            hovertemplate='Year: %{x}<br>Wells drilled: %{y}<extra></extra>',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yearly['Year'],
            y=yearly['rolling_avg'],
            mode='lines',
            name='5-year trend',
            line=dict(color='#b66d5d', width=2.4),
            hovertemplate='Year: %{x}<br>5-year trend: %{y:.1f}<extra></extra>',
        )
    )

    fig.add_annotation(
        x=peak_row['Year'],
        y=peak_row['count'],
        text=f"Peak · {int(peak_row['Year'])}",
        showarrow=True,
        arrowhead=2,
        ax=18,
        ay=-34,
        font=dict(family='DM Mono', size=10, color='#f1e4df'),
        bgcolor='rgba(182,109,93,0.14)',
        bordercolor='rgba(182,109,93,0.38)',
        borderpad=6,
        arrowcolor='rgba(182,109,93,0.48)',
    )

    fig.update_layout(
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Sora', color='#a8adc2', size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        bargap=0.14,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8f95ac'),
        ),
        hoverlabel=dict(
            bgcolor='rgba(15,17,22,0.96)',
            bordercolor='rgba(255,255,255,0.12)',
            font=dict(color='#eef2fb', family='Sora'),
        ),
    )
    fig.update_xaxes(
        title=None,
        showgrid=False,
        tickfont=dict(color='#6f758b'),
        linecolor='rgba(255,255,255,0.10)',
        showline=True,
        zeroline=False,
        tickmode='linear',
        dtick=5,
    )
    fig.update_yaxes(
        title=None,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        tickfont=dict(color='#6f758b'),
        linecolor='rgba(255,255,255,0.10)',
        showline=True,
        zeroline=False,
        rangemode='tozero',
    )
    return fig


def build_content_breakdown_chart(df):
    content = (
        df['Content']
        .fillna('Unknown')
        .replace({'': 'Unknown'})
        .value_counts()
        .reset_index()
    )
    content.columns = ['Content', 'count']
    content = content.sort_values('count', ascending=True).reset_index(drop=True)
    total = max(content['count'].sum(), 1)
    content['share'] = content['count'] / total * 100
    colors = build_gradient('#34486e', '#c8d4f0', len(content))

    fig = go.Figure(
        go.Bar(
            x=content['count'],
            y=content['Content'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.08)', width=0.6),
            ),
            customdata=np.stack([content['share']], axis=-1),
            text=[f"{count} · {share:.1f}%" for count, share in zip(content['count'], content['share'])],
            textposition='outside',
            cliponaxis=False,
            hovertemplate='Content: %{y}<br>Wells: %{x}<br>Share: %{customdata[0]:.1f}%<extra></extra>',
        )
    )

    fig.update_layout(
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Sora', color='#a8adc2', size=11),
        margin=dict(l=0, r=48, t=10, b=0),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='rgba(15,17,22,0.96)',
            bordercolor='rgba(255,255,255,0.12)',
            font=dict(color='#eef2fb', family='Sora'),
        ),
    )
    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        tickfont=dict(color='#6f758b'),
        linecolor='rgba(255,255,255,0.10)',
        showline=True,
        zeroline=False,
        rangemode='tozero',
    )
    fig.update_yaxes(
        title=None,
        tickfont=dict(color='#d7dde9'),
        showgrid=False,
        linecolor='rgba(255,255,255,0.10)',
        showline=False,
    )
    return fig


def render_overview_page(df, stats):
    render_overview_hero()
    st.markdown("<div style='margin: 26px 0 0;'></div>", unsafe_allow_html=True)
    render_metric_cards(stats)
    st.markdown("<div style='margin: 48px 0 0;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Exploration wells mapped by SODIR coordinates</div>', unsafe_allow_html=True)
    try:
        field_outlines = load_field_outlines()
    except Exception:
        field_outlines = {"geojson": {"type": "FeatureCollection", "features": []}, "labels": [], "outlines": []}
    render_well_map(df, field_outlines)

    render_oil_price_section()

    st.markdown("<div style='margin: 34px 0 0;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Exploration activity snapshots</div>', unsafe_allow_html=True)
    st.markdown(
        "<p class='subtle-copy' style='margin: 8px 0 20px;'>A paired view of how drilling activity evolved over time and how the content mix of Norwegian shelf wells is distributed across the current SODIR exploration dataset.</p>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1.22, 0.96], gap='large')

    with left:
        st.markdown('<div class="section-label">Wells drilled per year</div>', unsafe_allow_html=True)
        st.markdown(
            "<p class='subtle-copy' style='margin: 8px 0 14px;'>Bars show annual exploration intensity, while the copper trend line smooths the long-cycle rhythm across the shelf.</p>",
            unsafe_allow_html=True
        )
        fig = build_wells_drilled_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Well content breakdown</div>', unsafe_allow_html=True)
        st.markdown(
            "<p class='subtle-copy' style='margin: 8px 0 14px;'>A ranked horizontal layout makes it easier to compare outcomes than the old pie view, especially when categories are close in size.</p>",
            unsafe_allow_html=True
        )
        fig = build_content_breakdown_chart(df)
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
                request_page_change('Well Detail')
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
            request_page_change('Wells')
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

pending_page = st.session_state.pop('pending_page', None)
if pending_page in {"Overview", "Operators", "Wells", "Well Detail"}:
    st.session_state['page'] = pending_page

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
