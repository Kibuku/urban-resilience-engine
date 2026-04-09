"""
Phase 4 — Streamlit Dashboard: Interactive Nairobi Risk Map.

Run: streamlit run src/phase4_deploy/app.py
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
import plotly.express as px
import joblib
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_PROCESSED, MODELS_DIR, NAIROBI_CENTER


# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Resilience Engine — Nairobi",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ Urban Resilience Engine")
st.markdown("### Predicting Extreme Weather Risk Across Nairobi")


# ── Load Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    grid = gpd.read_parquet(DATA_PROCESSED / "nairobi_grid_full.parquet")
    X    = pd.read_parquet(DATA_PROCESSED / "X_features.parquet")
    model = joblib.load(MODELS_DIR / "xgboost_risk.joblib")
    grid["risk_prob"] = model.predict_proba(X)[:, 1]
    return grid


try:
    grid = load_data()
except FileNotFoundError:
    st.error("⚠️ Run Phases 1–3 first to generate data and train the model.")
    st.stop()


# ── Column aliases ───────────────────────────────────────────────────
_building_col = (
    "building_density_per_km2" if "building_density_per_km2" in grid.columns
    else "building_count"
)
_poverty_col = "poverty_index" if "poverty_index" in grid.columns else "poverty_rate_percent"


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Controls")

risk_threshold = st.sidebar.slider(
    "Risk Threshold", 0.0, 1.0, 0.5, 0.05,
    help="Hexagons above this probability are flagged as high-risk"
)

grid["flagged"] = (grid["risk_prob"] >= risk_threshold).astype(int)
n_flagged = grid["flagged"].sum()
n_total   = len(grid)

st.sidebar.metric("Total Grid Cells", n_total)
st.sidebar.metric("High-Risk Cells",  n_flagged,
                  delta=f"{n_flagged / n_total * 100:.1f}%")

LAYERS = {
    "Risk Probability":   ("risk_prob",              "RdYlGn_r", "Risk (0–1)"),
    "Poverty Index":      (_poverty_col,              "OrRd",     "Poverty index"),
    "Flood Risk":         ("flood_risk_percent",      "Blues",    "Flood risk %"),
    "Heat Vulnerability": ("heat_vulnerability_index","YlOrRd",   "Heat vuln."),
    "Building Density":   (_building_col,             "Purples",  "Bldgs / km²"),
    "NDVI (Wet Season)":  ("ndvi_wet",                "Greens",   "NDVI"),
}

show_layer = st.sidebar.selectbox("Map Layer", list(LAYERS.keys()))
active_col, palette, legend_caption = LAYERS[show_layer]

map_opacity = st.sidebar.slider("Fill Opacity", 0.2, 1.0, 0.75, 0.05)
show_borders = st.sidebar.checkbox("Show hex borders", value=True)


# ── Colourmap ────────────────────────────────────────────────────────
PALETTE_COLOURS = {
    "RdYlGn_r": ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"],
    "OrRd":     ["#fff7ec", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"],
    "Blues":    ["#f7fbff", "#c6dbef", "#9ecae1", "#6baed6", "#2171b5", "#08306b"],
    "YlOrRd":   ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
    "Purples":  ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f", "#3f007d"],
    "Greens":   ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c", "#00441b"],
}

vmin = float(grid[active_col].min())
vmax = float(grid[active_col].max())

colormap = cm.LinearColormap(
    colors=PALETTE_COLOURS[palette],
    vmin=vmin,
    vmax=vmax,
    caption=f"{show_layer}  ({legend_caption})",
)


# ── Build Folium map ─────────────────────────────────────────────────
st.subheader(f"🗺️ Nairobi — {show_layer}")

m = folium.Map(
    location=list(NAIROBI_CENTER),
    zoom_start=12,
    tiles="CartoDB positron",
    prefer_canvas=True,        # WebGL canvas renderer — much smoother
)

# Serialise only the columns we need (faster than full GeoDataFrame)
tooltip_cols = ["h3_id", "risk_prob", active_col,
                _poverty_col, "flood_risk_percent", "ndvi_wet"]
tooltip_cols = [c for c in dict.fromkeys(tooltip_cols) if c in grid.columns]

geo_subset = grid[tooltip_cols + ["geometry"]].copy()

folium.GeoJson(
    data=geo_subset.__geo_interface__,
    name=show_layer,
    style_function=lambda feat: {
        "fillColor":   colormap(feat["properties"].get(active_col, vmin)),
        "fillOpacity": map_opacity,
        "color":       "#555555" if show_borders else "none",
        "weight":      0.4 if show_borders else 0,
    },
    highlight_function=lambda feat: {
        "fillOpacity": min(map_opacity + 0.2, 1.0),
        "weight":      2,
        "color":       "#222222",
    },
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_cols,
        aliases=[
            "Hex ID", "Risk prob.", show_layer,
            "Poverty idx", "Flood risk %", "NDVI",
        ][:len(tooltip_cols)],
        localize=True,
        sticky=True,
        style="font-size:12px; font-family:monospace;",
    ),
).add_to(m)

# Flagged-cell outline layer (threshold ring)
flagged_gdf = grid[grid["flagged"] == 1][["geometry"]].copy()
if len(flagged_gdf):
    folium.GeoJson(
        data=flagged_gdf.__geo_interface__,
        name="Flagged (high-risk)",
        style_function=lambda _: {
            "fillColor": "none",
            "fillOpacity": 0,
            "color": "#cc0000",
            "weight": 1.8,
        },
        tooltip="⚠️ High-risk cell",
    ).add_to(m)

colormap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, use_container_width=True, height=540, returned_objects=[])


# ── Summary Charts ───────────────────────────────────────────────────
st.subheader("📊 Risk Distribution")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        grid, x="risk_prob", nbins=40, color="flagged",
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        title="Risk Probability Distribution",
        labels={"risk_prob": "Predicted Risk Probability", "flagged": "Flagged"},
        opacity=0.85,
    )
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold {risk_threshold:.2f}")
    fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, width='stretch')

with col2:
    grid["_drain_size"] = grid["drainage_count"] + 1 if "drainage_count" in grid.columns else 1
    fig = px.scatter(
        grid,
        x=_building_col, y="risk_prob",
        color=_poverty_col,
        size="_drain_size",
        color_continuous_scale="RdYlGn_r",
        opacity=0.7,
        title="Risk vs Building Density  (size = drainage coverage)",
        labels={
            _building_col: "Building density (per km²)",
            "risk_prob":   "Risk probability",
            _poverty_col:  "Poverty index",
        },
        hover_data=["flood_risk_percent", "ndvi_wet"] if "flood_risk_percent" in grid.columns else [],
    )
    fig.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig, width='stretch')


# ── Executive Summary ────────────────────────────────────────────────
st.subheader("📝 Executive Summary for Decision-Makers")

st.markdown(f"""
**For the Governor's Office:**

Our model identified **{n_flagged} out of {n_total} neighbourhood zones**
({n_flagged / n_total * 100:.1f}%) across Nairobi as high-risk for flood and
heat stress events at the current threshold of **{risk_threshold:.0%}**.

**Key drivers of risk** (from SHAP analysis):
- 🚰 Low drainage infrastructure coverage — single strongest predictor
- 🏘️ High building density without green buffers
- 🌿 Vegetation loss between seasons (NDVI seasonal decline)
- 💸 Higher poverty levels — correlated with informal settlements in flood-prone areas

**Recommended interventions** (prioritised by cost-effectiveness):
1. **Drainage upgrades** in high-density, low-drainage hex zones — estimated
   to reduce flood risk by 15–25 % per zone
2. **Green corridor restoration** along Nairobi River, Mathare River, and
   Ngong River tributaries
3. **Early warning systems** targeted at the {n_flagged} flagged zones,
   prioritising Mathare, Korogocho, and Kibera sub-locations

⚠️ **Equity note:** See the Bias Audit Report — lower-income areas are
flagged more frequently, partly reflecting genuine vulnerability but
warranting careful policy interpretation before resource allocation.
""")
