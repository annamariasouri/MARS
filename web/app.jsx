// MARS Dashboard — main app
const { useState, useMemo } = React;
const { LineChart, ScatterChart, Sparkline, RiskStrip, fmtNum, fmtDate, RegionMap } = window;

// Tweak defaults — host can rewrite this block.
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "ocean",
  "accent": "teal",
  "showGrid": true,
  "kpiStyle": "card"
}/*EDITMODE-END*/;

const REGIONS = {
  thermaikos: { title: "Thermaikos (Greece)", short: "Thermaikos", country: "Greece", bbox: [40.2, 40.7, 22.5, 23.0] },
  peiraeus:   { title: "Piraeus (Greece)",     short: "Piraeus",    country: "Greece", bbox: [37.9, 38.1, 23.5, 23.8] },
  limassol:   { title: "Limassol (Cyprus)",    short: "Limassol",   country: "Cyprus", bbox: [34.6, 34.8, 33.0, 33.2] },
};

const ENV_VARS = [
  { key: "chl",    label: "Chlorophyll-a",   unit: "mg/m³",  color: "var(--teal)" },
  { key: "thetao", label: "Temperature θ",   unit: "°C",      color: "#ff8a64" },
  { key: "so",     label: "Salinity",        unit: "PSU",     color: "#a5b6ff" },
  { key: "nh4",    label: "Ammonium NH₄",    unit: "µmol/L",  color: "#f2b13a" },
  { key: "no3",    label: "Nitrate NO₃",     unit: "µmol/L",  color: "#85e0a5" },
  { key: "po4",    label: "Phosphate PO₄",   unit: "µmol/L",  color: "#c084fc" },
];

const META = () => window.MARS_META || {};
const VALIDATION = () => {
  const rows = META().validation;
  if (rows && rows.length) return rows.filter(r => !r.is_total);
  return [
    { region: "Thermaikos", n: 60984, rmse: 0.1283, mae: 0.0629, r2: 0.9185 },
    { region: "Piraeus",    n: 90387, rmse: 0.0070, mae: 0.0040, r2: 0.9742 },
    { region: "Limassol",   n: 8712,  rmse: 0.0051, mae: 0.0031, r2: 0.9085 },
  ];
};
const VALIDATION_TOTAL = () => {
  const rows = META().validation;
  const total = rows && rows.find(r => r.is_total);
  if (total) return total;
  const v = VALIDATION();
  return {
    region: "Overall",
    n: v.reduce((s, r) => s + (r.n || 0), 0),
    rmse: 0.0794,
    mae: 0.0264,
    r2: 0.9449,
  };
};
const BLOOM_METRICS = () => {
  const rows = META().bloom_metrics;
  if (rows && rows.length) return rows;
  return [
    { region: "Thermaikos", threshold: 1.00, prevalence: 12.4, precision: 0.825, recall: 0.849, f1: 0.837 },
    { region: "Piraeus",    threshold: 0.50, prevalence: 0.0,  precision: 0,     recall: 0,     f1: 0 },
    { region: "Limassol",   threshold: 0.30, prevalence: 0.0,  precision: 0,     recall: 0,     f1: 0 },
  ];
};

function riskLevel(pct) {
  if (pct == null || isNaN(pct)) return 'low';
  if (pct <= 33) return 'low';
  if (pct <= 66) return 'med';
  return 'high';
}
function riskLabel(level) {
  return level === 'high' ? 'High' : level === 'med' ? 'Moderate' : 'Low';
}

// ====================================================================

function Sidebar({ activeNav, setActiveNav }) {
  const [logoVariant, setLogoVariant] = React.useState(
    () => localStorage.getItem('mars-logo-variant') || 'pulse'
  );
  React.useEffect(() => {
    const onStorage = () => setLogoVariant(localStorage.getItem('mars-logo-variant') || 'pulse');
    window.addEventListener('storage', onStorage);
    // also poll periodically since same-tab localStorage changes don't fire 'storage'
    const t = setInterval(onStorage, 800);
    return () => { window.removeEventListener('storage', onStorage); clearInterval(t); };
  }, []);

  const navItems = [
    { id: 'overview', label: 'Overview' },
    { id: 'forecast', label: "Today's Forecast" },
    { id: 'trends',   label: 'Environmental Trends' },
    { id: 'accuracy', label: 'Model Accuracy' },
    { id: 'methods',  label: 'Methodology' },
  ];
  const tools = [
    { id: 'data',     label: 'Data sources' },
    { id: 'logo',     label: 'Logo & identity', href: 'logo.html' },
    { id: 'export',   label: 'Export report' },
    { id: 'diag',     label: 'Diagnostics' },
  ];
  return (
    <aside className="sidebar">
      <div className="brand">
        {window.MarsLogo
          ? <window.MarsLogo variant={logoVariant} size={40} />
          : <div className="brand-mark">M</div>}
        <div className="brand-text">
          <div className="brand-title">MARS</div>
          <div className="brand-sub">Marine Autonomous Risk System</div>
        </div>
      </div>

      <div className="nav-section">
        <div className="nav-label">Dashboard</div>
        {navItems.map(n => (
          <div key={n.id}
            className={`nav-item ${activeNav === n.id ? 'active' : ''}`}
            onClick={() => setActiveNav(n.id)}>
            <span className="nav-dot" />
            <span>{n.label}</span>
          </div>
        ))}
      </div>

      <div className="nav-section">
        <div className="nav-label">Tools</div>
        {tools.map(n => n.href ? (
          <a key={n.id} className="nav-item" href={n.href}
             style={{textDecoration:'none'}}>
            <span className="nav-dot" />
            <span>{n.label}</span>
          </a>
        ) : (
          <div key={n.id} className="nav-item" onClick={() => {}}>
            <span className="nav-dot" />
            <span>{n.label}</span>
          </div>
        ))}
      </div>

      <div className="sidebar-meta">
        <strong>PhD research prototype</strong><br/>
        Annamaria Souri · UNIC<br/>
        Forecasting harmful algal blooms in the Eastern Mediterranean using daily Copernicus Marine data and a Random Forest model.

        <div className="copernicus-badge">
          <div className="copernicus-mark"></div>
          <div className="copernicus-text">
            <strong>Copernicus Marine</strong><br/>
            Daily ingest · v2026.05
          </div>
        </div>
      </div>
    </aside>
  );
}

// ====================================================================

function Hero({ regionData, region, asOf, theme, onToggleTheme }) {
  return (
    <div className="hero">
      <div>
        <div className="hero-eyebrow">Live · Updated daily</div>
        <h1 className="hero-title">
          Red-tide risk forecast for the <em>Eastern Mediterranean</em>
        </h1>
        <p className="hero-sub">
          A daily marine bloom outlook combining Copernicus environmental observations with a trained
          Random Forest model. Selected basin: <strong style={{color:'var(--text)'}}>{REGIONS[region].title}</strong>.
        </p>
      </div>
      <div className="hero-meta">
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
          title={theme === 'light' ? 'Switch to ocean (dark) mode' : 'Switch to daylight (light) mode'}
          aria-label="Toggle theme">
          <span className="theme-toggle-track">
            <span className="theme-toggle-icon sun">
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none">
                <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.4"/>
                {[0,45,90,135,180,225,270,315].map(a => (
                  <line key={a} x1="8" y1="1.5" x2="8" y2="3"
                    stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"
                    transform={`rotate(${a} 8 8)`} />
                ))}
              </svg>
            </span>
            <span className="theme-toggle-icon moon">
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none">
                <path d="M 12.5 9.5 A 5.5 5.5 0 1 1 6.5 3.5 A 4 4 0 0 0 12.5 9.5 Z"
                      fill="currentColor"/>
              </svg>
            </span>
            <span className="theme-toggle-thumb" data-theme={theme} />
          </span>
          <span className="theme-toggle-label">{theme === 'light' ? 'Daylight' : 'Ocean'}</span>
        </button>
        <div className="hero-meta-row" style={{marginTop:6}}>Forecast as of</div>
        <div className="hero-meta-val">{asOf}</div>
        <div className="hero-meta-row" style={{marginTop:6}}>Model</div>
        <div className="hero-meta-val">rf_chl · v2026.05</div>
      </div>
    </div>
  );
}

// ====================================================================

function MapSection({ region, setRegion, regionSummaries, coverage }) {
  const cov = coverage || {};
  return (
    <div className="map-card">
      <div className="map-pane">
        <RegionMap
          active={region}
          onSelect={setRegion}
          regions={REGIONS}
          summaries={regionSummaries}
        />
      </div>
      <div className="region-list">
        <div className="region-list-header">
          <div className="region-list-title">Monitored basins</div>
          <span style={{fontFamily:'var(--font-mono)', fontSize:10, color:'var(--text-muted)'}}>
            3 / 3 ACTIVE
          </span>
        </div>
        {Object.entries(REGIONS).map(([k, v]) => {
          const sum = regionSummaries[k];
          const level = sum?.level || 'low';
          return (
            <div key={k} className={`region-row ${region===k?'active':''}`}
                 onClick={() => setRegion(k)}>
              <span className="region-row-color" style={{
                background: level==='high' ? 'var(--danger)' : level==='med' ? 'var(--warn)' : 'var(--ok)',
                boxShadow: `0 0 8px ${level==='high' ? 'var(--danger)' : level==='med' ? 'var(--warn)' : 'var(--ok)'}`,
              }} />
              <div className="region-row-content">
                <div className="region-row-name">{v.short}</div>
                <div className="region-row-meta">
                  {v.country} · CHL {fmtNum(sum?.chl, 3)} mg/m³
                </div>
              </div>
              <span className={`badge ${level}`}>{riskLabel(level)}</span>
            </div>
          );
        })}
        <div style={{padding:'14px 18px', marginTop:'auto', borderTop:'1px solid var(--line)'}}>
          <div style={{fontSize:10.5, color:'var(--text-muted)', letterSpacing:1.2, textTransform:'uppercase', fontWeight:600, marginBottom:6}}>Coverage window</div>
          <div style={{fontFamily:'var(--font-mono)', fontSize:12, color:'var(--text)'}}>
            {cov.start && cov.end ? `${cov.start} → ${cov.end}` : '—'}
          </div>
          <div style={{fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-muted)', marginTop:2}}>
            {cov.forecast_days != null ? `${cov.forecast_days} forecast days` : '—'} · {cov.basin_count || 3} basins
          </div>
        </div>
      </div>
    </div>
  );
}

// ====================================================================

function KpiCard({ label, value, unit, sub, info, sparkData, sparkColor, badge, badgeLevel }) {
  return (
    <div className="kpi">
      <div className="kpi-label">
        <span>{label}</span>
        {info && (
          <details className="kpi-info" style={{position:'relative'}}>
            <summary style={{listStyle:'none', cursor:'help'}}>?</summary>
          </details>
        )}
      </div>
      {badge ? (
        <div style={{marginTop:4}}>
          <span className={`badge ${badgeLevel}`}>{badge}</span>
        </div>
      ) : (
        <div className="kpi-value">
          {value}
          {unit && <span className="kpi-value-unit">{unit}</span>}
        </div>
      )}
      {sub && <div className="kpi-sub">{sub}</div>}
      {sparkData && (
        <div className="kpi-spark">
          <Sparkline values={sparkData} color={sparkColor || 'var(--teal)'} height={32} />
        </div>
      )}
    </div>
  );
}

function KpiRow({ region, summary, env }) {
  const recentChl = env.slice(-14).map(d => d.chl);
  const recentTemp = env.slice(-14).map(d => d.thetao);

  return (
    <>
      <div className="section-head">
        <h2 className="section-title">{REGIONS[region].title} · today</h2>
        <div className="section-sub">RF · 90th-percentile adaptive threshold</div>
      </div>

      <div className="kpi-grid">
        <KpiCard
          label="Predicted CHL"
          value={fmtNum(summary.chl, 3)}
          unit="mg/m³"
          sub={
            <>
              <span className={`kpi-trend ${summary.chlTrend > 0 ? 'up' : summary.chlTrend < 0 ? 'down' : 'flat'}`}>
                {summary.chlTrend > 0 ? '↑' : summary.chlTrend < 0 ? '↓' : '→'} {Math.abs(summary.chlTrend*100).toFixed(1)}%
              </span>
              <span>vs 7-day avg</span>
            </>
          }
          sparkData={recentChl} sparkColor="var(--teal)"
        />

        <KpiCard
          label="Continuous risk"
          value=""
          badge={`${riskLabel(riskLevel(summary.risk_score))} · ${fmtNum(summary.risk_score, 1)}%`}
          badgeLevel={riskLevel(summary.risk_score)}
          sub={<span>How close to threshold today</span>}
        />

        <KpiCard
          label="Adaptive threshold"
          value={fmtNum(summary.threshold, 3)}
          unit="mg/m³"
          sub={<span>P90 over rolling history</span>}
          sparkData={summary.thresholdSeries}
          sparkColor="var(--warn)"
        />

        <KpiCard
          label="Likelihood · 7d"
          value=""
          badge={`${riskLabel(riskLevel(summary.rec7))} · ${fmtNum(summary.rec7, 0)}%`}
          badgeLevel={riskLevel(summary.rec7)}
          sub={<span>{summary.risk7}/7 risk days projected</span>}
        />

        <KpiCard
          label="Likelihood · 30d"
          value=""
          badge={`${riskLabel(riskLevel(summary.rec30))} · ${fmtNum(summary.rec30, 0)}%`}
          badgeLevel={riskLevel(summary.rec30)}
          sub={<span>{summary.risk30}/30 risk days projected</span>}
        />

        <KpiCard
          label="Sea θ · today"
          value={fmtNum(summary.temp, 1)}
          unit="°C"
          sub={
            <>
              <span className={`kpi-trend ${summary.tempTrend > 0 ? 'up' : 'down'}`}
                style={{color: summary.tempTrend > 0 ? 'var(--warn)' : 'var(--ok)'}}>
                {summary.tempTrend > 0 ? '↑' : '↓'} {Math.abs(summary.tempTrend).toFixed(2)}°
              </span>
              <span>vs 7-day avg</span>
            </>
          }
          sparkData={recentTemp} sparkColor="#ff8a64"
        />
      </div>
    </>
  );
}

// ====================================================================

function TabsBlock({ region, forecast, env, accuracy, summary }) {
  const [tab, setTab] = useState('forecast');
  const tabs = [
    { id: 'forecast', label: "Today's Forecast" },
    { id: 'trends',   label: 'Environmental Trends' },
    { id: 'accuracy', label: 'Model Accuracy' },
  ];

  return (
    <div className="tabs-card">
      <div className="tabs-header">
        {tabs.map(t => (
          <div key={t.id} className={`tab ${tab===t.id?'active':''}`} onClick={() => setTab(t.id)}>
            {t.label}
          </div>
        ))}
      </div>
      <div className="tab-body">
        {tab === 'forecast' && <ForecastTab region={region} forecast={forecast} env={env} summary={summary} />}
        {tab === 'trends'   && <TrendsTab region={region} env={env} />}
        {tab === 'accuracy' && <AccuracyTab region={region} accuracy={accuracy} />}
      </div>
    </div>
  );
}

function ForecastTab({ region, forecast, env, summary }) {
  const last30 = forecast.slice(-30);
  const last7 = forecast.slice(-7);
  const envLast30 = env.slice(-30);

  return (
    <div style={{display:'flex', flexDirection:'column', gap:18}}>
      <div className="chart-grid-2">
        <div className="chart">
          <div className="chart-header">
            <div className="chart-title">CHL · last 7 days · observed env-history</div>
            <div className="chart-meta">mg/m³</div>
          </div>
          <LineChart
            data={env.slice(-7)} x="date" height_unit="mg/m³"
            ys={[{key:'chl', label:'CHL', color:'var(--teal)'}]}
            height={200}
          />
        </div>
        <div className="chart">
          <div className="chart-header">
            <div className="chart-title">CHL · last 30 days · with prediction overlay</div>
            <div className="chart-meta">mg/m³</div>
          </div>
          <LineChart
            data={envLast30} x="date" height_unit="mg/m³"
            ys={[{key:'chl', label:'CHL', color:'var(--teal)'}]}
            threshold={summary.threshold}
            height={200}
          />
        </div>
      </div>

      <div className="chart">
        <div className="chart-header">
          <div className="chart-title">Forecast log · daily predicted CHL vs adaptive threshold</div>
          <div className="chart-meta">{last30.length} days · {REGIONS[region].short}</div>
        </div>
        <LineChart
          data={last30} x="date" height_unit="mg/m³"
          ys={[
            { key: 'predicted_chl', label: 'Predicted CHL', color: 'var(--depth)' },
            { key: 'threshold',     label: 'Threshold',     color: 'var(--warn)' },
          ]}
          areaFill={false}
          height={240}
        />
        <div style={{display:'flex', alignItems:'center', justifyContent:'space-between', marginTop:14}}>
          <div>
            <div style={{fontSize:10.5, color:'var(--text-muted)', letterSpacing:1.2, textTransform:'uppercase', fontWeight:600, marginBottom:6}}>
              Daily risk state · last 30 days
            </div>
            <RiskStrip values={last30.map(d => d.predicted_chl)} threshold={summary.threshold} />
          </div>
          <div style={{display:'flex', gap:14, alignItems:'center', fontSize:11, color:'var(--text-muted)'}}>
            <span style={{display:'inline-flex', alignItems:'center', gap:5}}>
              <span style={{display:'inline-block', width:10, height:10, borderRadius:2, background:'var(--ok-soft)'}}></span> Low
            </span>
            <span style={{display:'inline-flex', alignItems:'center', gap:5}}>
              <span style={{display:'inline-block', width:10, height:10, borderRadius:2, background:'var(--warn-soft)'}}></span> Approach
            </span>
            <span style={{display:'inline-flex', alignItems:'center', gap:5}}>
              <span style={{display:'inline-block', width:10, height:10, borderRadius:2, background:'var(--danger-soft)'}}></span> Bloom
            </span>
          </div>
        </div>
      </div>

      <details className="expander">
        <summary>How does the model produce these numbers?</summary>
        <div className="expander-body">
          Daily Copernicus Marine fields (nutrients, temperature, salinity, surface chlorophyll) are
          ingested per basin. We compute lagged values, rolling averages, nutrient ratios and anomalies,
          then feed them to a retrained Random Forest (<code>rf_chl_retrained.pkl</code>). The model emits
          a daily CHL prediction per grid cell, summarised to the basin median. A bloom-risk flag fires
          when the prediction crosses an adaptive threshold — the 90th percentile of predicted CHL over
          the rolling history — which keeps false positives down in naturally productive water.
        </div>
      </details>
    </div>
  );
}

function TrendsTab({ region, env }) {
  const defaults = ['chl', 'thetao'];
  const [selected, setSelected] = useState(defaults);
  const toggle = k => setSelected(s => s.includes(k) ? s.filter(x => x!==k) : [...s, k]);

  return (
    <div>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:8}}>
        <div>
          <div style={{fontSize:13, fontWeight:600, color:'var(--text)'}}>Environmental drivers · {REGIONS[region].short}</div>
          <div style={{fontSize:11.5, color:'var(--text-muted)', marginTop:2}}>
            Last 30 days · Copernicus reanalysis · click variables to toggle
          </div>
        </div>
        <div className="section-sub">{env.length} obs</div>
      </div>

      <div className="var-chips">
        {ENV_VARS.map(v => (
          <button key={v.key}
                  className={`var-chip ${selected.includes(v.key)?'on':''}`}
                  onClick={() => toggle(v.key)}>
            <span className="var-chip-dot" style={{background: v.color}}></span>
            {v.label}
          </button>
        ))}
      </div>

      <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(380px, 1fr))', gap:14}}>
        {selected.map(k => {
          const v = ENV_VARS.find(x => x.key===k);
          return (
            <div key={k} className="chart">
              <div className="chart-header">
                <div className="chart-title">{v.label}</div>
                <div className="chart-meta">{v.unit}</div>
              </div>
              <LineChart
                data={env} x="date"
                ys={[{key:k, label:v.label, color:v.color}]}
                height={180}
                height_unit={v.unit}
              />
            </div>
          );
        })}
      </div>

      {selected.length === 0 && (
        <div style={{padding:40, textAlign:'center', color:'var(--text-muted)', fontSize:12}}>
          Select a variable to plot.
        </div>
      )}
    </div>
  );
}

function AccuracyTab({ region, accuracy }) {
  const observed = accuracy.filter(d => d.observed_chl != null);
  const recent = [...accuracy].sort((a,b) => new Date(b.target_date) - new Date(a.target_date)).slice(0, 12);

  return (
    <div style={{display:'flex', flexDirection:'column', gap:18}}>
      <div style={{display:'grid', gridTemplateColumns:'minmax(0, 1fr) minmax(0, 1fr)', gap:14}}>
        <div className="chart">
          <div className="chart-header">
            <div className="chart-title">Predicted vs Observed · time series</div>
            <div className="chart-meta">{REGIONS[region].short}</div>
          </div>
          <LineChart
            data={accuracy} x="target_date"
            ys={[
              { key:'predicted_chl', label:'Predicted', color:'var(--depth)' },
              { key:'observed_chl',  label:'Observed',  color:'var(--teal)' },
            ]}
            height={220} areaFill={false}
            height_unit="mg/m³"
          />
        </div>
        <div className="chart">
          <div className="chart-header">
            <div className="chart-title">Predicted vs Observed · scatter</div>
            <div className="chart-meta">n = {observed.length}</div>
          </div>
          <ScatterChart data={observed} xKey="predicted_chl" yKey="observed_chl" height={220} />
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'minmax(0, 1fr) minmax(0, 1fr)', gap:14}}>
        <div className="chart" style={{padding:0}}>
          <div className="chart-header" style={{padding:'14px 18px 10px'}}>
            <div className="chart-title">Validation metrics · 2023–2025</div>
            <div className="chart-meta">RMSE · MAE · R²</div>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Region</th><th className="num">n</th><th className="num">RMSE</th>
                <th className="num">MAE</th><th className="num">R²</th>
              </tr>
            </thead>
            <tbody>
              {VALIDATION().map(r => (
                <tr key={r.region}>
                  <td style={{color:'var(--text)'}}>{r.region}</td>
                  <td className="num">{r.n != null ? r.n.toLocaleString() : '—'}</td>
                  <td className="num">{r.rmse != null ? r.rmse.toFixed(4) : '—'}</td>
                  <td className="num">{r.mae != null ? r.mae.toFixed(4) : '—'}</td>
                  <td className="num" style={{color: r.r2 > 0.9 ? 'var(--ok)' : 'var(--text)'}}>
                    {r.r2 != null ? r.r2.toFixed(4) : '—'}
                  </td>
                </tr>
              ))}
              {(() => {
                const t = VALIDATION_TOTAL();
                return (
              <tr className="total">
                <td>Overall</td>
                <td className="num">{t.n != null ? t.n.toLocaleString() : '—'}</td>
                <td className="num">{t.rmse != null ? t.rmse.toFixed(4) : '—'}</td>
                <td className="num">{t.mae != null ? t.mae.toFixed(4) : '—'}</td>
                <td className="num">{t.r2 != null ? t.r2.toFixed(4) : '—'}</td>
              </tr>
                );
              })()}
            </tbody>
          </table>
        </div>

        <div className="chart" style={{padding:0}}>
          <div className="chart-header" style={{padding:'14px 18px 10px'}}>
            <div className="chart-title">Bloom detection · precision/recall</div>
            <div className="chart-meta">basin thresholds</div>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Region</th><th className="num">Thr</th><th className="num">Prev.</th>
                <th className="num">Prec.</th><th className="num">Rec.</th><th className="num">F1</th>
              </tr>
            </thead>
            <tbody>
              {BLOOM_METRICS().map(r => (
                <tr key={r.region}>
                  <td style={{color:'var(--text)'}}>{r.region}</td>
                  <td className="num">{r.threshold != null ? r.threshold.toFixed(2) : '—'}</td>
                  <td className="num">{r.prevalence != null ? `${r.prevalence.toFixed(1)}%` : '—'}</td>
                  <td className="num">{r.precision.toFixed(3)}</td>
                  <td className="num">{r.recall.toFixed(3)}</td>
                  <td className="num" style={{color: r.f1 > 0.7 ? 'var(--ok)' : r.f1 > 0 ? 'var(--text)' : 'var(--text-muted)'}}>
                    {r.f1.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="chart" style={{padding:0}}>
        <div className="chart-header" style={{padding:'14px 18px 10px'}}>
          <div className="chart-title">Recent forecasts · {REGIONS[region].short}</div>
          <div className="chart-meta">12 most recent · ground truth pending after lag</div>
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th>Date</th>
              <th className="num">Predicted</th>
              <th className="num">Observed</th>
              <th className="num">Error</th>
              <th className="num">|Error|</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((r, i) => {
              const hasObs = r.observed_chl != null;
              return (
                <tr key={i}>
                  <td style={{color:'var(--text)'}}>{r.target_date}</td>
                  <td className="num">{fmtNum(r.predicted_chl, 3)}</td>
                  <td className="num">{hasObs ? fmtNum(r.observed_chl, 3) : '—'}</td>
                  <td className="num">{hasObs ? fmtNum(r.err, 3) : '—'}</td>
                  <td className="num">{hasObs ? fmtNum(r.abs_err, 3) : '—'}</td>
                  <td style={{fontFamily:'var(--font-ui)'}}>
                    {hasObs
                      ? <span style={{color: Math.abs(r.err) < 0.2 ? 'var(--ok)' : 'var(--warn)'}}>
                          {Math.abs(r.err) < 0.2 ? '✓ within tol.' : '○ off tol.'}
                        </span>
                      : <span style={{color:'var(--text-muted)'}}>● pending obs</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ====================================================================

function App() {
  const [t, setTweak] = window.useTweaks(TWEAK_DEFAULTS);

  // Apply theme
  React.useEffect(() => {
    const root = document.documentElement;
    if (t.theme === 'light') {
      root.classList.add('theme-light');
      root.removeAttribute('data-theme');
    } else {
      root.classList.remove('theme-light');
      root.removeAttribute('data-theme');
    }
    const map = {
      teal:  { teal: '#4dd0c5', depth: '#5c93ff' },
      cyan:  { teal: '#67e8f9', depth: '#3b82f6' },
      kelp:  { teal: '#a3e635', depth: '#0ea5e9' },
      coral: { teal: '#fb7185', depth: '#38bdf8' },
    };
    const c = map[t.accent] || map.teal;
    root.style.setProperty('--teal', c.teal);
    root.style.setProperty('--depth', c.depth);
  }, [t.theme, t.accent]);

  const [region, setRegion] = useState('thermaikos');
  const [activeNav, setActiveNav] = useState('overview');

  const data = window.MARS_DATA;
  const forecast = data[region].forecast;
  const env = data[region].env;
  const accuracy = data[region].accuracy;

  // Build summary per region
  const summaries = useMemo(() => {
    const out = {};
    Object.keys(REGIONS).forEach(r => {
      const fc = data[r].forecast;
      const e = data[r].env;
      const last = fc[fc.length - 1];
      const recentEnvChl = e.slice(-7).map(d => d.chl).filter(v => !isNaN(v));
      const recentEnvTemp = e.slice(-7).map(d => d.thetao).filter(v => !isNaN(v));
      const meanChl7 = recentEnvChl.length ? recentEnvChl.reduce((a,b)=>a+b,0)/recentEnvChl.length : 0;
      const meanTemp7 = recentEnvTemp.length ? recentEnvTemp.reduce((a,b)=>a+b,0)/recentEnvTemp.length : 0;
      const currChl = e[e.length-1]?.chl;
      const currTemp = e[e.length-1]?.thetao;

      const last7 = fc.slice(-7);
      const last30 = fc.slice(-30);
      const risk7 = last7.filter(d => d.predicted_chl >= d.threshold).length;
      const risk30 = last30.filter(d => d.predicted_chl >= d.threshold).length;

      out[r] = {
        chl: currChl,
        chlTrend: meanChl7 ? (currChl - meanChl7) / meanChl7 : 0,
        temp: currTemp,
        tempTrend: currTemp - meanTemp7,
        threshold: last.threshold,
        risk_score: last.risk_score,
        rec7: (risk7 / 7) * 100,
        rec30: (risk30 / 30) * 100,
        risk7, risk30,
        thresholdSeries: last30.map(d => d.threshold),
        level: riskLevel(last.risk_score),
      };
    });
    return out;
  }, [data]);

  const summary = summaries[region];
  const asOf = forecast[forecast.length - 1]?.date || '—';

  return (
    <div className="app">
      <Sidebar activeNav={activeNav} setActiveNav={setActiveNav} />

      <main className="main">
        <Hero
          regionData={REGIONS[region]}
          region={region}
          asOf={asOf}
          theme={t.theme}
          onToggleTheme={() => setTweak('theme', t.theme === 'light' ? 'ocean' : 'light')}
        />

        <div>
          <div className="section-head" style={{marginBottom:10}}>
            <h2 className="section-title">Regional outlook · Eastern Mediterranean</h2>
            <div className="section-sub">3 basins · click any to focus</div>
          </div>
          <MapSection
            region={region}
            setRegion={setRegion}
            regionSummaries={summaries}
            coverage={{
              start: META().coverage_start,
              end: META().coverage_end,
              forecast_days: META().forecast_days,
              basin_count: META().basin_count,
            }}
          />
        </div>

        <KpiRow region={region} summary={summary} env={env} />

        <TabsBlock region={region} forecast={forecast} env={env} accuracy={accuracy} summary={summary} />

        <div className="footer">
          <div className="footer-trail">
            <span>© 2026 MARS · Research prototype</span>
            <span>•</span>
            <span>Data: Copernicus Marine Service</span>
            <span>•</span>
            <span>Model: rf_chl_retrained v2026.05</span>
          </div>
          <div style={{fontFamily:'var(--font-mono)', fontSize:10.5, color:'var(--text-muted)', letterSpacing:0.5}}>
            UNIC · PhD thesis project · A. Souri
          </div>
        </div>
      </main>

      {/* Tweaks panel */}
      {window.TweaksPanel && (
        <window.TweaksPanel title="Tweaks">
          <window.TweakSection label="Theme">
            <window.TweakRadio
              label="Mode"
              value={t.theme}
              onChange={v => setTweak('theme', v)}
              options={[
                { value: 'ocean', label: 'Ocean' },
                { value: 'light', label: 'Daylight' },
              ]}
            />
            <window.TweakRadio
              label="Accent"
              value={t.accent}
              onChange={v => setTweak('accent', v)}
              options={[
                { value: 'teal',  label: 'Teal' },
                { value: 'cyan',  label: 'Cyan' },
                { value: 'kelp',  label: 'Kelp' },
                { value: 'coral', label: 'Coral' },
              ]}
            />
          </window.TweakSection>
        </window.TweaksPanel>
      )}
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
