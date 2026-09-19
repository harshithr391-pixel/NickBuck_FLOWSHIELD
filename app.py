"""
Urban Flood Risk Simulator — single-file version
===================================================
Everything (terrain generation, simulation engine, scenario presets,
visualization, and the Streamlit dashboard) lives in this one file so
the whole project can be run from a single script.

Pipeline: Rainfall & Terrain Input -> Water-Level Simulation ->
          Flood Progression Visualizer -> Risk Classification ->
          Early Warning Metrics

Run with:
    pip install streamlit numpy pandas plotly
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ===========================================================================
# 1. TERRAIN / CITY GENERATION
# ===========================================================================

def _smooth(grid, iterations=2):
    """Simple box-blur smoothing implemented with plain NumPy (no SciPy
    dependency) so the terrain doesn't look like pure random static."""
    g = grid.copy()
    for _ in range(iterations):
        padded = np.pad(g, 1, mode="edge")
        g = (
            padded[1:-1, 1:-1] * 4
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 8.0
    return g


def generate_city(rows, cols, seed=42):
    """Builds a synthetic but plausible city grid: a connected grid of
    regions, each with elevation, drainage capacity, population density,
    and an initial water level. Adjacent cells are implicitly connected
    in a 4-neighbor (von Neumann) graph, used later for water movement.
    """
    rng = np.random.default_rng(seed)

    # Elevation: smoothed noise plus a gentle city-wide tilt, so one corner
    # behaves like a natural low-lying river basin that water drains toward.
    raw = rng.normal(size=(rows, cols))
    elevation = _smooth(raw, iterations=4)
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-9)
    row_grad, col_grad = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    tilt = 0.4 * (row_grad + col_grad) / 2
    elevation = 0.7 * elevation + 0.3 * tilt
    elevation = elevation * 100  # scale to 0-100 "meters"

    # Drainage capacity: generally better on higher, more developed ground;
    # noisier and weaker in low-lying areas.
    base_drainage = 20 - 0.12 * elevation
    drainage_noise = rng.normal(0, 2.0, size=(rows, cols))
    drainage_capacity = np.clip(base_drainage + drainage_noise, 3, 22)

    # Population density: people cluster on flatter, lower ground near the
    # implied river/city center.
    density_noise = rng.normal(0, 300, size=(rows, cols))
    population_density = np.clip(6000 - 45 * elevation + density_noise, 200, 9000)

    water_level = np.zeros((rows, cols))

    return {
        "elevation": elevation,
        "drainage_capacity": drainage_capacity,
        "population_density": population_density,
        "water_level": water_level,
    }


# ===========================================================================
# 2. SIMULATION ENGINE
# ===========================================================================

class FloodSimulation:
    """Implements:
        Water Level(t+1) = Water Level(t) + Rainfall - Drainage Output
                            ± Net Inflow/Outflow
    plus gravity/terrain-based lateral water movement, risk
    classification (Safe/Warning/Critical), and estimated
    time-to-critical per region.
    """

    def __init__(self, elevation, drainage_capacity, population_density,
                 base_critical_mm=80.0, warning_ratio=0.6, flow_coefficient=0.15):
        self.elevation = elevation
        self.drainage_capacity = drainage_capacity
        self.population_density = population_density
        self.rows, self.cols = elevation.shape
        self.flow_coefficient = flow_coefficient

        # Regions with weaker drainage infrastructure tip into "critical"
        # at a lower absolute water depth than well-drained regions.
        mean_drainage = drainage_capacity.mean()
        self.critical_threshold = np.clip(
            base_critical_mm * (mean_drainage / drainage_capacity), 30, 220
        )
        self.warning_threshold = self.critical_threshold * warning_ratio

    def _lateral_flow(self, water):
        """Moves water from higher effective-surface cells (elevation + water)
        to lower adjacent cells, capped so a cell can't empty in one step."""
        head = self.elevation + water
        outflow = np.zeros_like(water)
        inflow = np.zeros_like(water)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_head = np.roll(head, (-dr, -dc), axis=(0, 1))
            # Closed boundary: no wrap-around flow off the edge of the grid.
            if dr == -1:
                neighbor_head[-1, :] = head[-1, :]
            elif dr == 1:
                neighbor_head[0, :] = head[0, :]
            if dc == -1:
                neighbor_head[:, -1] = head[:, -1]
            elif dc == 1:
                neighbor_head[:, 0] = head[:, 0]

            diff = head - neighbor_head
            flow = np.clip(diff, 0, None) * self.flow_coefficient
            flow = np.minimum(flow, water / 4.0)
            outflow += flow
            inflow += np.roll(flow, (dr, dc), axis=(0, 1))

        return np.clip(water - outflow + inflow, 0, None)

    def run(self, duration_hours, rainfall_fn, drainage_efficiency=None, dt_hours=1.0):
        """Runs the simulation. Returns (history, rainfall_history).
        history has shape (steps+1, rows, cols); history[0] is the initial state."""
        if drainage_efficiency is None:
            drainage_efficiency = np.ones_like(self.elevation)

        steps = int(duration_hours / dt_hours)
        history = np.zeros((steps + 1, self.rows, self.cols))
        rainfall_history = np.zeros(steps + 1)
        water = np.zeros_like(self.elevation)
        history[0] = water

        for t in range(steps):
            hour = t * dt_hours
            rainfall = rainfall_fn(hour) * dt_hours
            water = water + rainfall
            drainage_output = np.minimum(water, self.drainage_capacity * drainage_efficiency * dt_hours)
            water = np.clip(water - drainage_output, 0, None)
            water = self._lateral_flow(water)
            history[t + 1] = water
            rainfall_history[t + 1] = rainfall

        return history, rainfall_history

    def classify_risk(self, water_grid):
        """0 = Safe, 1 = Warning, 2 = Critical."""
        risk = np.zeros_like(water_grid, dtype=int)
        risk[water_grid >= self.warning_threshold] = 1
        risk[water_grid >= self.critical_threshold] = 2
        return risk

    def affected_population(self, water_grid, include_warning=True):
        risk = self.classify_risk(water_grid)
        mask = risk >= (1 if include_warning else 2)
        return float(self.population_density[mask].sum())

    def time_to_critical(self, history):
        """Per-cell estimated hours until first reaching critical.
        np.inf means never reached and no rising trend was detected."""
        steps = history.shape[0]
        eta = np.full((self.rows, self.cols), np.inf)

        for r in range(self.rows):
            for c in range(self.cols):
                series = history[:, r, c]
                crit = self.critical_threshold[r, c]
                reached = np.where(series >= crit)[0]
                if len(reached) > 0:
                    eta[r, c] = reached[0]
                else:
                    n = min(6, steps)
                    if n >= 2:
                        y = series[-n:]
                        x = np.arange(n)
                        slope, _intercept = np.polyfit(x, y, 1)
                        if slope > 1e-6:
                            remaining = crit - series[-1]
                            eta[r, c] = (steps - 1) + remaining / slope
                        # flat/falling trend -> leave as np.inf (not projected to flood)
        return eta


# ===========================================================================
# 3. SCENARIO PRESETS
# ===========================================================================

def constant_rainfall(intensity_mm_per_hr):
    return lambda hour: intensity_mm_per_hr


def storm_curve(peak_intensity, duration_hours, peak_hour_fraction=0.4):
    """A triangular storm profile: rainfall rises to a peak then tapers off."""
    peak_hour = duration_hours * peak_hour_fraction

    def fn(hour):
        if hour <= peak_hour:
            return peak_intensity * (hour / max(peak_hour, 1e-6))
        remaining = duration_hours - peak_hour
        return peak_intensity * max(0.0, 1 - (hour - peak_hour) / max(remaining, 1e-6))

    return fn


PRESETS = {
    "Normal Rainfall": {
        "rainfall_fn": lambda duration: constant_rainfall(6.0),
        "drainage_efficiency": "full",
        "description": "Steady light-to-moderate rain (6 mm/hr) with drainage operating normally.",
    },
    "Heavy Rainfall": {
        "rainfall_fn": lambda duration: storm_curve(35.0, duration),
        "drainage_efficiency": "full",
        "description": "A storm peaking at 35 mm/hr partway through the event; drainage operating normally.",
    },
    "Complete Drainage Failure": {
        "rainfall_fn": lambda duration: constant_rainfall(10.0),
        "drainage_efficiency": "none",
        "description": "Moderate rainfall (10 mm/hr) with the stormwater system entirely offline, city-wide.",
    },
    "Blocked Drainage Channel": {
        "rainfall_fn": lambda duration: constant_rainfall(10.0),
        "drainage_efficiency": "custom",
        "description": "Moderate rainfall (10 mm/hr) with drainage disabled only in selected regions.",
    },
}


def build_drainage_efficiency(mode, shape, blocked_mask=None):
    """mode: 'full' -> drainage works everywhere; 'none' -> disabled everywhere;
    'custom' -> disabled only where blocked_mask is True."""
    if mode == "full":
        return np.ones(shape)
    if mode == "none":
        return np.zeros(shape)
    if mode == "custom":
        eff = np.ones(shape)
        if blocked_mask is not None:
            eff[blocked_mask] = 0.0
        return eff
    raise ValueError(f"Unknown drainage efficiency mode: {mode}")


# ===========================================================================
# 4. VISUALIZATION (Plotly figure builders)
# ===========================================================================

RISK_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c"]  # Safe, Warning, Critical
RISK_LABELS = ["Safe", "Warning", "Critical"]


def risk_heatmap(risk_grid, elevation, title="Flood Risk"):
    rows, cols = risk_grid.shape
    hover = [
        [
            f"Region ({r},{c})<br>Elevation: {elevation[r, c]:.1f}m<br>Risk: {RISK_LABELS[risk_grid[r, c]]}"
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=risk_grid,
            colorscale=[[0, RISK_COLORS[0]], [0.5, RISK_COLORS[1]], [1, RISK_COLORS[2]]],
            zmin=0,
            zmax=2,
            showscale=False,
            text=hover,
            hoverinfo="text",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(showticklabels=False),
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def water_level_heatmap(water_grid, title="Water Level (mm)"):
    fig = go.Figure(
        data=go.Heatmap(z=water_grid, colorscale="Blues", showscale=True, xgap=2, ygap=2)
    )
    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(showticklabels=False),
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def metrics_timeseries(hours, values_dict, ylabel):
    fig = go.Figure()
    for name, values in values_dict.items():
        fig.add_trace(go.Scatter(x=hours, y=values, mode="lines+markers", name=name))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title=ylabel,
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig


# ===========================================================================
# 5. STREAMLIT DASHBOARD
# ===========================================================================

st.set_page_config(page_title="Urban Flood Simulator", layout="wide")

st.title("🌧️ Urban Flood Risk Simulator")
st.caption("Grid-based rainfall, drainage and terrain-driven flood progression model.")

# ---------------------------------------------------------------------------
# Sidebar: city + simulation configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("City Setup")
    grid_size = st.slider("Grid size (N x N)", 6, 20, 12)
    seed = st.number_input("Terrain seed", 0, 9999, 42)

    st.header("Simulation")
    duration = st.slider("Simulation duration (hours)", 6, 48, 24)
    base_critical = st.slider("Base critical water depth (mm)", 40, 200, 80, step=10)
    flow_coefficient = st.slider(
        "Lateral flow strength", 0.0, 0.4, 0.15, step=0.01,
        help="How readily water moves from higher to lower ground each hour.",
    )

city = generate_city(grid_size, grid_size, seed=int(seed))
sim = FloodSimulation(
    city["elevation"], city["drainage_capacity"], city["population_density"],
    base_critical_mm=base_critical, flow_coefficient=flow_coefficient,
)
total_population = float(city["population_density"].sum())
cell_options = [f"{r},{c}" for r in range(grid_size) for c in range(grid_size)]

tab_sim, tab_compare = st.tabs(["📍 Live Simulation", "⚖️ Scenario Comparison"])

# ---------------------------------------------------------------------------
# Tab 1: Live single-scenario simulation
# ---------------------------------------------------------------------------
with tab_sim:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        preset_name = st.selectbox("Scenario preset", list(PRESETS.keys()), key="sim_preset")
    with col_b:
        blocked_cells = []
        if PRESETS[preset_name]["drainage_efficiency"] == "custom":
            blocked_cells = st.multiselect(
                "Regions with blocked drainage (row,col)", cell_options,
                default=cell_options[: max(1, grid_size // 3)], key="sim_blocked",
            )

    preset = PRESETS[preset_name]
    rainfall_fn = preset["rainfall_fn"](duration)
    blocked_mask = None
    if blocked_cells:
        blocked_mask = np.zeros((grid_size, grid_size), dtype=bool)
        for cell in blocked_cells:
            r, c = map(int, cell.split(","))
            blocked_mask[r, c] = True
    drainage_eff = build_drainage_efficiency(
        preset["drainage_efficiency"], city["elevation"].shape, blocked_mask
    )

    st.info(preset["description"])

    history, rainfall_history = sim.run(duration, rainfall_fn, drainage_eff)
    eta_grid = sim.time_to_critical(history)

    hour = st.slider("Hour", 0, duration, min(6, duration), key="sim_hour")
    water_now = history[hour]
    risk_now = sim.classify_risk(water_now)

    safe_n = int((risk_now == 0).sum())
    warn_n = int((risk_now == 1).sum())
    crit_n = int((risk_now == 2).sum())
    affected_pop = sim.affected_population(water_now, include_warning=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Safe regions", safe_n)
    m2.metric("Warning regions", warn_n)
    m3.metric("Critical regions", crit_n)
    m4.metric(
        "Est. affected population", f"{affected_pop:,.0f}",
        help=f"{affected_pop / total_population:.1%} of ~{total_population:,.0f} total",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            risk_heatmap(risk_now, city["elevation"], f"Risk at hour {hour}"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            water_level_heatmap(water_now, f"Water level (mm) at hour {hour}"),
            use_container_width=True,
        )

    st.subheader("Flood progression over time")
    hours_axis = np.arange(duration + 1)
    safe_series, warn_series, crit_series, pop_series = [], [], [], []
    for t in range(duration + 1):
        r = sim.classify_risk(history[t])
        safe_series.append((r == 0).sum())
        warn_series.append((r == 1).sum())
        crit_series.append((r == 2).sum())
        pop_series.append(sim.affected_population(history[t]))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            metrics_timeseries(
                hours_axis, {"Safe": safe_series, "Warning": warn_series, "Critical": crit_series},
                "Number of regions",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            metrics_timeseries(hours_axis, {"Affected population": pop_series}, "People"),
            use_container_width=True,
        )

    st.subheader("Early warning: regions closest to critical")
    rows_list = []
    for r in range(grid_size):
        for c in range(grid_size):
            eta = eta_grid[r, c]
            eta_label = f"{eta:.1f} h" if np.isfinite(eta) else "Not projected to reach critical"
            rows_list.append({
                "Region": f"({r},{c})",
                "Elevation (m)": round(float(city["elevation"][r, c]), 1),
                "Drainage capacity (mm/hr)": round(float(city["drainage_capacity"][r, c]), 1),
                "Population": int(city["population_density"][r, c]),
                "Current water (mm)": round(float(water_now[r, c]), 1),
                "Estimated time to critical": eta_label,
                "_eta_sort": eta,
            })
    warning_df = pd.DataFrame(rows_list).sort_values("_eta_sort").drop(columns="_eta_sort").head(10)
    st.dataframe(warning_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: Side-by-side scenario comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.write("Compare two scenarios side by side using the same city and terrain.")
    colA, colB = st.columns(2)
    with colA:
        preset_a = st.selectbox("Scenario A", list(PRESETS.keys()), index=0, key="cmp_a")
    with colB:
        default_b_index = 1 if len(PRESETS) > 1 else 0
        preset_b = st.selectbox("Scenario B", list(PRESETS.keys()), index=default_b_index, key="cmp_b")

    def run_scenario(name):
        p = PRESETS[name]
        r_fn = p["rainfall_fn"](duration)
        eff = build_drainage_efficiency(p["drainage_efficiency"], city["elevation"].shape)
        hist, _ = sim.run(duration, r_fn, eff)
        return hist

    history_a = run_scenario(preset_a)
    history_b = run_scenario(preset_b)

    hour_cmp = st.slider("Hour", 0, duration, min(12, duration), key="cmp_hour")

    col1, col2 = st.columns(2)
    with col1:
        risk_a = sim.classify_risk(history_a[hour_cmp])
        st.plotly_chart(
            risk_heatmap(risk_a, city["elevation"], f"{preset_a} — hour {hour_cmp}"),
            use_container_width=True,
        )
        st.metric("Affected population", f"{sim.affected_population(history_a[hour_cmp]):,.0f}")
    with col2:
        risk_b = sim.classify_risk(history_b[hour_cmp])
        st.plotly_chart(
            risk_heatmap(risk_b, city["elevation"], f"{preset_b} — hour {hour_cmp}"),
            use_container_width=True,
        )
        st.metric("Affected population", f"{sim.affected_population(history_b[hour_cmp]):,.0f}")

    st.subheader("Affected population over time")
    hours_axis = np.arange(duration + 1)
    pop_a = [sim.affected_population(history_a[t]) for t in range(duration + 1)]
    pop_b = [sim.affected_population(history_b[t]) for t in range(duration + 1)]
    st.plotly_chart(
        metrics_timeseries(hours_axis, {preset_a: pop_a, preset_b: pop_b}, "People"),
        use_container_width=True,
    )

    st.subheader("Summary at selected hour")

    def summarize(name, hist):
        risk = sim.classify_risk(hist[hour_cmp])
        return {
            "Scenario": name,
            "Safe": int((risk == 0).sum()),
            "Warning": int((risk == 1).sum()),
            "Critical": int((risk == 2).sum()),
            "Affected population": f"{sim.affected_population(hist[hour_cmp]):,.0f}",
        }

    summary_df = pd.DataFrame([summarize(preset_a, history_a), summarize(preset_b, history_b)])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
