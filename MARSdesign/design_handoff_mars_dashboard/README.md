# Handoff: MARS — Marine Autonomous Risk System dashboard

## Overview

MARS is a marine bloom-risk forecasting dashboard for the Eastern Mediterranean.
The system combines daily Copernicus Marine environmental observations with a trained
Random Forest model to produce daily chlorophyll (CHL) forecasts and bloom-risk
likelihoods for three monitored locations (Thermaikos, Piraeus, Limassol).

This bundle contains the **design prototype** for the dashboard.

## About the Design Files

The files in this bundle are **design references created in HTML/JSX (via Babel in
the browser)** — they are prototypes showing intended look, behavior, and data flow.
They are **not production code to copy directly**.

The task is to **recreate these HTML designs in your existing codebase's
environment** (React, Vue, SvelteKit, Next.js, etc.) using its established patterns,
component libraries, state management, and data sources — or, if no environment
exists yet, to choose the most appropriate framework and implement the designs there.

The prototype works as static files (open `MARS Dashboard.html` in a browser),
which is useful for review, but the JSX is transpiled in-browser via Babel —
**replace this with a real build step** (Vite + React, Next.js, etc.) in production.

## Fidelity

**High-fidelity (hifi)**: pixel-perfect mockups with final colors, typography,
spacing, interactions, and chart treatments. Recreate the UI pixel-perfectly,
matching tokens to your codebase's design system where possible.

## Architecture

- **Two themes**: `daylight` (default, light) and `ocean` (dark). Theme is toggled
  via a class `theme-light` on `<html>`. CSS custom properties (`--bg`, `--surface`,
  `--text`, `--teal`, `--depth`, etc.) swap per theme.
- **Single-page layout**: sticky topbar → hero → regional outlook (map + locations
  list) → KPI grid → tabbed analytics block (Today's Forecast / Environmental
  Trends / Model Accuracy) → footer.
- **Charts**: hand-rolled SVG line, scatter, sparkline, and risk-strip components.
  In a production rewrite, replace with your codebase's preferred chart library
  (Recharts, Visx, ECharts, D3, etc.) but preserve the visual treatment
  (gradients, axis style, threshold lines).
- **Map**: Leaflet 1.9 + CartoDB Voyager (light) / Esri Dark Gray (ocean) basemaps.
  Custom DOM markers with concentric pulse rings.

## Screens / Views

### 1. Topbar (sticky)
**Purpose**: brand identity, project attribution, theme toggle.

**Layout**: grid `auto 1fr auto`, items centered, 14px vertical padding, 36px
horizontal padding.

**Components**:
- **MARS brand** (left): 38px logo (`MarsLogo` component, variant `pulse`) +
  "MARS" wordmark (14px, weight 600, letter-spacing 0.5px) + subtitle
  "Marine Autonomous Risk System" (10.5px uppercase, muted). Border-right
  divider, padded.
- **Credit sentence**: "Part of the PhD research of Annamaria Souri ·
  UNIVERSITY OF NICOSIA". Body in `--font-ui` 12.5px muted; the institution
  is rendered in a small caps mono treatment (`--font-mono`, 11px, letter-spacing
  0.8px, uppercase, slightly darker tone).
- **Theme toggle** (right): 34px pill with a 50×26 track containing a sun
  (filled gold `#f59e0b`, 8 rays) and a crescent moon (slate). A 22×22 thumb
  with a teal→navy gradient and inset highlight slides between the two icons.
  The inactive icon dims to ~45–50% opacity. Label "DAYLIGHT" / "OCEAN" in
  mono caps to the right.

### 2. Hero
**Purpose**: orient the viewer — what is this, what's the latest update.

**Layout**: grid `1fr auto`, items end-aligned, 24px gap. In light mode,
flat (no card box) with a 48×3 cyan→navy accent bar at top-left and a
bottom border separator.

**Components**:
- **Eyebrow**: "LIVE · UPDATED DAILY" (11px uppercase, teal, letter-spacing
  2px) with a 6px circular dot prefix glowing teal.
- **Title**: "Red-tide risk forecast for the *Eastern Mediterranean*" —
  `--font-display` (Newsreader serif), 38px, weight 500, letter-spacing -1.1px,
  the location italicized in teal.
- **Subtitle**: single-line copy explaining the system + currently selected
  location in bold. `--font-ui` 14px, muted. Truncates with ellipsis at narrow
  widths (<1100px wraps normally).
- **Meta block** (right): right-aligned stack of "FORECAST AS OF" + date
  (mono), "MODEL" + version (mono). Small caps labels.

### 3. Regional outlook
**Purpose**: pick a location to focus on; see all three at a glance.

**Layout**: card with grid `1fr 300px`, map on left, locations list on right.

**Map** (left):
- Leaflet container, full bleed inside its column.
- Light: CartoDB Voyager tiles + Voyager labels overlay.
- Ocean: Esri World_Dark_Gray + World_Dark_Gray_Reference; a CSS filter
  (`sepia(1) hue-rotate(195deg) saturate(2.2) brightness(0.55) contrast(1.05)`)
  tints the dark grey toward deep marine blue.
- **No bbox rectangles** — only circular pin markers per location.
- **Custom pin**: concentric ring + dot, colored by risk level (var `--danger`,
  `--warn`, `--ok`). Active pin scales up and pulses. Hover reveals a small
  rounded label with the location name + current CHL value.
- **Auto-zoom**: when a location is selected (via list click or pin click),
  the map `flyToBounds(bbox, { padding: [80,80], maxZoom: 10, duration: 0.9 })`.
  First load skips this so all three are visible.
- **Custom controls**: bottom-left title card ("EASTERN MEDITERRANEAN · 19°–36° E"
  + "3 monitored locations"); top-right zoom (+/−/⤢ reset) buttons styled
  to match the card.

**Locations list** (right):
- Header: "MONITORED LOCATIONS" + "3 / 3 ACTIVE" (mono).
- 3 rows. Each row has:
  - A **status dot** on the left, colored by risk level with a glowing box-shadow.
  - **Name row**: country flag SVG (Greece: blue/white stripes + cross hoist;
    Cyprus: white + orange map shape + green olive branches), then the location
    name, then a **trend chip** (↑/↓/→ arrow + percent change vs the 7-day mean,
    red for up = worsening, green for down = improving, muted for flat).
  - **Meta row**: "<Country> · CHL 0.XXX mg/m³" + an inline **14-day sparkline**
    on the right (~64px wide) colored by risk level.
  - A **risk badge** on the right: "HIGH" / "WATCH" / "LOW" pill in
    soft-color background matching the level.
- Footer: "COVERAGE WINDOW" + ISO date range (mono), "97 forecast days · 3 locations".
- Click a row → setRegion(k) → triggers map auto-zoom + updates the rest of the
  dashboard to that location.

### 4. KPI grid (6 cards)
**Purpose**: at-a-glance current status of the selected location.

**Layout**: grid 6 columns ≥ 1440px, 3 columns ≥ 720px, else 2 columns. 14px gap.

**Cards** (in order):
1. **Predicted CHL** — large numeric (e.g. `0.350 mg/m³`), trend "↓ 0.2% vs 7-day
   avg", sparkline of recent CHL in deep blue.
2. **Continuous risk** — risk badge (HIGH 84.1%), sub "How close to threshold today".
3. **Adaptive threshold** — `0.444 mg/m³`, sub "P90 over rolling history",
   sparkline of threshold history in amber.
4. **Likelihood · 7d** — risk badge for next-7-day projection, sub "{N}/7 risk
   days projected". Has a hover **tooltip** (`?`) with full definition.
5. **Likelihood · 30d** — same pattern, 30-day horizon. Tooltip.
6. **Sea T° · today** — temperature value with delta vs 7-day mean, sparkline.

**KPI card visual**: 1px border, 14px radius, 18px padding. In light mode each
card carries a thin cyan-to-transparent accent stripe across the top edge
(`::before`, `top:0; left:16px; right:16px; height:2px`) for the "smart data
card" feel. Subtle shadow.

### 5. Tabs block
**Purpose**: drill into time-series and accuracy data.

**Tabs**: Today's Forecast (default), Environmental Trends, Model Accuracy.

**Today's Forecast tab**:
- **CHL · last 7 days** chart (env-history, single-series area+line).
- **CHL · last 30 days · with prediction overlay** chart (env-history line
  + predicted overlay).
- **Forecast log · daily predicted CHL vs adaptive threshold** chart (two-series:
  predicted in deep blue, threshold in amber).
- Below: **daily risk state strip** (last 30 days, color-coded blocks for
  bloom/approach/low).
- An expander ("how the forecast is computed") with a paragraph of methodology.

**Environmental Trends tab**:
- Multi-line chart of environmental drivers (chl, temp, salinity, etc.) over
  the last 30 days. Variables toggleable via legend chips.

**Model Accuracy tab**:
- **Predicted vs Observed · time series** chart: predicted in deep blue, observed
  in **warm amber `#e89a2b`** (deliberate contrast — in the deep-blue palette,
  using teal for observed makes the two lines blend; amber separates them clearly).
- **Predicted vs Observed · scatter** chart: actual vs predicted points with
  a perfect-fit reference line.
- **Validation metrics · 2023–2025** table (RMSE / MAE / R² per location).
  Has a hover **tooltip (?)** explaining metrics.
- **Bloom detection · precision/recall** table (precision / recall / F1 per
  location). Has a hover **tooltip (?)** explaining metrics.
- **Recent forecasts** table (12 most recent rows: date, predicted, observed,
  error, |error|).

### 6. Footer
- Trail: "© 2026 MARS · Research prototype" · "Data: Copernicus Marine Service"
  · "Model: rf_chl_retrained v2026.05".
- Right side: "UNIC · PhD thesis project · A. Souri" in mono small caps.

## Interactions & Behavior

- **Region selection**: clicking a row in the locations list OR a map pin
  changes the selected region. All downstream (hero subtitle, KPI cards, tabs)
  update. Map auto-zooms to bbox.
- **Theme toggle**: instant switch via class on `<html>`. The map detects via
  `MutationObserver` on `<html>` class and swaps tile URLs without reinitializing
  the map.
- **Tweaks panel** (`?tweaks=1` or via host toolbar): floating panel exposes
  *Theme · Mode* (radio Ocean/Daylight) and *Theme · Accent* (radio: Deep blue,
  Teal, Cyan, Kelp, Coral). Default accent: `deepblue`. Each accent has a
  different `--teal` and `--depth` pair per theme — `deepblue` in light is
  `#0891b2` / `#0c4a6e`; in ocean it's `#22d3ee` / `#7dd3fc`.
- **Tooltips (?)**: 13px circle-Q icon, on hover/focus reveals a 260px white
  rounded bubble above the icon (with a tail), fades in/out.
- **Hover states**:
  - Region rows: subtle bg shift.
  - Map pins: scale + label reveal.
  - Theme toggle: border darkens.
  - Tooltip icons: color → teal.

## State Management

- `tweaks` (theme, accent) — persists via `__edit_mode_set_keys` host message
  and is rewritten into the `EDITMODE-BEGIN/END` JSON block in `app.jsx`.
  In your codebase, replace with your preferred theme/preferences store
  (localStorage, Zustand, Pinia, etc.).
- `region` — currently selected location key (`thermaikos`, `peiraeus`, `limassol`).
- `activeTab` — `forecast` | `trends` | `accuracy`.
- `logoVariant` — read from localStorage `mars-logo-variant`. Allows user to
  pick a logo variant on the `MARS Logo.html` page; the dashboard observes the
  storage key and updates.

## Data Sources

- **Forecast & environmental data**: bundled in `data.js` as `window.MARS_DATA`,
  keyed by region. Each region has:
  - `forecast`: daily rows with `date`, `predicted_chl`, `threshold`, `risk_score`.
  - `env`: daily Copernicus env rows (`chl`, `thetao` sea temp, `so` salinity,
    `no3`, `po4`, etc.).
  - `accuracy`: rows of `{ target_date, predicted_chl, observed_chl, err, abs_err }`.
- **Sample raw data**: `data_samples/` contains the source CSVs/JSON the
  prototype's data was extracted from.
- In production, replace `window.MARS_DATA` with a fetch from your backend
  (FastAPI / Flask serving Copernicus + RF predictions, or static JSON in
  object storage).

## Design Tokens

### Colors — Ocean (default / dark)
- `--bg: #061325`
- `--bg-2: #08182e`
- `--surface: #0c2140`
- `--surface-2: #0f2950`
- `--line: rgba(140, 180, 220, 0.10)`
- `--line-strong: rgba(140, 180, 220, 0.18)`
- `--text: #e8eef7`
- `--text-dim: #9bb0c8`
- `--text-muted: #6c819a`
- Accent (deepblue): `--teal: #22d3ee`, `--depth: #7dd3fc`
- Status: `--ok: #34c896`, `--warn: #f2b13a`, `--danger: #ff5e7e`

### Colors — Daylight (light)
- `--bg: #f6f8fa`
- `--bg-2: #ebeef2`
- `--surface: #ffffff`
- `--surface-2: #f9fafb`
- `--line: rgba(15, 23, 42, 0.07)`
- `--line-strong: rgba(15, 23, 42, 0.14)`
- `--text: #0f172a`
- `--text-dim: #334155`
- `--text-muted: #64748b`
- Accent (deepblue): `--teal: #0891b2`, `--depth: #0c4a6e`
- Status: `--ok: #047857`, `--warn: #b45309`, `--danger: #be123c`
- Body gradient: `linear-gradient(180deg, #fafbfd 0%, #e9edf2 100%)`

### Typography
- `--font-ui`: `'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif`
- `--font-mono`: `'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace`
- `--font-display`: `'Newsreader', 'Source Serif Pro', Georgia, serif`

### Radius & Shadow
- `--radius: 14px` (cards), `--radius-sm: 8px` (chips/badges)
- Card shadows (light): `0 0 0 1px rgba(15,23,42,0.02), 0 8px 24px -16px rgba(12,74,110,0.20)`
- Tooltip shadow: `0 0 0 1px rgba(15,23,42,0.02), 0 12px 32px -12px rgba(15,23,42,0.28)`

### Spacing (rough scale)
- Page padding: 36px horizontal, 28px top, 60px bottom
- Card padding: 18px (KPI), 14–18px (chart header)
- Gap (KPI grid): 14px
- Gap (sections): 24px

## Assets

- **Logo**: `logo.jsx` exports `<MarsLogo variant="pulse|depth|signal|basin" size={N} />`.
  All four variants are SVG-only (no raster assets). Default variant is `pulse`.
  Logo identity / variant chooser lives in `MARS Logo.html`.
- **Country flags**: inline SVG (`FlagIcon` component in `app.jsx`) — no external
  files. Greece and Cyprus are hand-drawn at 18×12 viewbox.
- **Favicon**: `favicon.png` (32×32).
- **Map tiles**: external CDNs (CartoDB Voyager + Esri ArcGIS). No auth keys
  required for either, but consider proxying through your own infrastructure
  if you expect high traffic.

## Files in this bundle

- `MARS Dashboard.html` — entry HTML (loads React + Babel + Leaflet, then the
  JSX files).
- `app.jsx` — main app: layout, sidebar→topbar, hero, region selector,
  KPI row, tabs block, tweaks panel mount. ~830 lines.
- `charts.jsx` — `LineChart`, `ScatterChart`, `Sparkline`, `RiskStrip`,
  formatters.
- `map.jsx` — Leaflet `RegionMap` web component (`window.RegionMap`).
- `data.js` — `window.MARS_DATA` payload (forecast + env + accuracy per region).
- `data_samples/` — raw CSV/JSON the prototype's bundled data was derived from.
- `logo.jsx` — `MarsLogo` SVG component (4 variants).
- `tweaks-panel.jsx` — host-integrated tweaks panel (you can drop this in your
  own theme preferences page or remove entirely).
- `styles.css` — all styling. Theme tokens in `:root` / `.theme-light` at the top.
- `MARS Logo.html` — logo variant chooser page (auxiliary, used during
  brand development).
- `favicon.png`

## Recommended re-implementation stack

If starting fresh:
- **Vite + React + TypeScript** for the app shell.
- **TanStack Query** for forecast data fetching.
- **Recharts** or **Visx** for charts (drop the hand-rolled SVG).
- **Leaflet 1.9** (or **MapLibre GL** for vector tiles) — keep the existing map
  approach.
- **Tailwind CSS** or **CSS Modules** — port the tokens from `styles.css` 1:1
  into your theme config.
- **Zustand** for `region` + `theme` + `accent` state (tiny, no Provider needed).

## Known caveats

- JSX is transpiled at runtime via Babel `@babel/standalone` — DO NOT ship this
  to production. Babel runtime transpilation is slow and unreliable for real
  apps.
- All chart components are hand-rolled SVG. They look great but lack accessibility
  affordances (aria labels, keyboard nav, screen-reader friendly tables). A real
  chart library handles this for you.
- The "Tools" tab/links in the original sidebar were removed during design
  iteration. If your app needs nav, plug in your own.
- The Tweaks panel relies on the host's `__edit_mode_*` postMessage protocol.
  Replace with your codebase's settings/preferences pattern (or omit entirely).
- Map tile URLs hard-code remote basemap providers (CartoDB, Esri ArcGIS). For
  production, consider self-hosting tiles or using a paid provider with SLA.
