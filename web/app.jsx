// MARS Dashboard — main app
const { useState, useMemo } = React;
const { LineChart, ScatterChart, Sparkline, RiskStrip, fmtNum, fmtDate, RegionMap } = window;

// Tweak defaults — host can rewrite this block.
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light",
  "accent": "deepblue",
  "showGrid": true,
  "kpiStyle": "card"
}/*EDITMODE-END*/;

const REGIONS = {
  thermaikos: { title: "Port of Thessaloniki (Greece)", short: "Thessaloniki", country: "Greece", bbox: [40.2, 40.7, 22.5, 23.0] },
  peiraeus:   { title: "Port of Piraeus (Greece)",      short: "Piraeus",    country: "Greece", bbox: [37.9, 38.1, 23.5, 23.8] },
  limassol:   { title: "Port of Limassol (Cyprus)",     short: "Limassol",   country: "Cyprus", bbox: [34.6, 34.8, 33.0, 33.2] },
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
    { region: "Thessaloniki", n: 60984, rmse: 0.1283, mae: 0.0629, r2: 0.9185 },
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
    { region: "Thessaloniki", threshold: 1.00, prevalence: 12.4, precision: 0.825, recall: 0.849, f1: 0.837 },
    { region: "Piraeus",    threshold: 0.50, prevalence: 0.0,  precision: 0,     recall: 0,     f1: 0 },
    { region: "Limassol",   threshold: 0.30, prevalence: 0.0,  precision: 0,     recall: 0,     f1: 0 },
  ];
};
const MODEL_LABEL = () => META().model_version || "rf_chl · v2026.05";
const ACCURACY_TOL = 0.2; // mg/m³
const ACCURACY_STATUS_TIP = `On target = predicted CHL within ${ACCURACY_TOL} mg/m³ of Copernicus CHL. Off target = larger error. Pending = no observation yet.`;

/** Share of forecast days over bloom threshold (7d / 30d). */
function outlookRiskLevel(pct) {
  if (pct == null || isNaN(pct)) return 'low';
  if (pct <= 33) return 'low';
  if (pct <= 66) return 'med';
  return 'high';
}
function outlookRiskLabel(pct) {
  if (pct == null || isNaN(pct)) return '—';
  if (pct <= 33) return 'Low';
  if (pct <= 66) return 'Medium';
  return 'High';
}

const RISK_MEDIUM_PCT = 75;
const RISK_HIGH_PCT = 90;

/** Today's bloom risk vs adaptive P90 threshold (Low → Alert). */
function bloomRiskTier(score, bloomFlag) {
  if (bloomFlag === 1 || bloomFlag === true) {
    return { key: 'alert', label: 'Alert', badgeLevel: 'high' };
  }
  if (score == null || isNaN(score)) {
    return { key: 'low', label: 'Low', badgeLevel: 'low' };
  }
  if (score >= 100) {
    return { key: 'alert', label: 'Alert', badgeLevel: 'high' };
  }
  if (score >= RISK_HIGH_PCT) {
    return { key: 'high', label: 'High', badgeLevel: 'high' };
  }
  if (score >= RISK_MEDIUM_PCT) {
    return { key: 'medium', label: 'Medium', badgeLevel: 'med' };
  }
  return { key: 'low', label: 'Low', badgeLevel: 'low' };
}

function bloomRiskSub(score, predicted, threshold) {
  const parts = [];
  if (score != null && !isNaN(score)) parts.push(`${fmtNum(score, 0)}% of threshold`);
  if (predicted != null && threshold != null) {
    parts.push(`${fmtNum(predicted, 3)} vs ${fmtNum(threshold, 3)} mg/m³`);
  }
  return parts.join(' · ') || 'Predicted CHL vs adaptive threshold';
}

function forecastAccuracyStatus(row) {
  const obs = row.observed_chl;
  const err = row.err;
  if (obs == null || obs === '' || isNaN(obs)) {
    return { key: 'pending', label: 'Pending', color: 'var(--text-muted)' };
  }
  if (err == null || isNaN(err)) {
    return { key: 'unknown', label: '—', color: 'var(--text-muted)' };
  }
  if (Math.abs(err) < ACCURACY_TOL) {
    return { key: 'on', label: 'On target', color: 'var(--ok)' };
  }
  return { key: 'off', label: 'Off target', color: 'var(--warn)' };
}

function mapPinLabel(levelKey) {
  const map = {
    low: 'Low',
    medium: 'Medium',
    med: 'Medium',
    high: 'High',
    alert: 'Alert',
  };
  return map[levelKey] || 'Low';
}

/** Pair each forecast day with Copernicus CHL from env-history (Environmental Trends). */
function mergeAccuracyWithEnv(accuracy, env) {
  const chlByDate = {};
  (env || []).forEach(d => {
    if (d.date != null && d.chl != null && !isNaN(d.chl)) chlByDate[d.date] = d.chl;
  });
  return (accuracy || []).map(row => {
    const date = row.target_date;
    let observed = row.observed_chl;
    if ((observed == null || observed === '') && chlByDate[date] != null) {
      observed = chlByDate[date];
    }
    const pred = row.predicted_chl;
    let err = row.err;
    let abs_err = row.abs_err;
    if (observed != null && pred != null && !isNaN(observed) && !isNaN(pred)) {
      err = pred - observed;
      abs_err = Math.abs(err);
    }
    return { ...row, observed_chl: observed, err, abs_err };
  });
}

// ====================================================================

function TopBar({ theme, onToggleTheme }) {
  const [logoVariant, setLogoVariant] = React.useState(
    () => localStorage.getItem('mars-logo-variant') || 'pulse'
  );
  React.useEffect(() => {
    const onStorage = () => setLogoVariant(localStorage.getItem('mars-logo-variant') || 'pulse');
    window.addEventListener('storage', onStorage);
    const t = setInterval(onStorage, 800);
    return () => { window.removeEventListener('storage', onStorage); clearInterval(t); };
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-brand" title="MARS — Marine Autonomous Risk System">
        {window.MarsLogo
          ? <window.MarsLogo variant={logoVariant} size={38} />
          : <div className="brand-mark">M</div>}
        <div className="brand-text">
          <div className="brand-title">MARS</div>
          <div className="brand-sub">Marine Autonomous Risk System</div>
        </div>
      </div>

      <div className="topbar-credit">
        <span className="topbar-credit-text">
          Part of the PhD research of Annamaria Souri · <span className="topbar-credit-inst">University of Nicosia</span>
        </span>
      </div>

      <div className="topbar-toggle-wrap">
        <button
          type="button"
          className="theme-toggle theme-toggle--topbar"
          onClick={onToggleTheme}
          title={theme === 'light' ? 'Switch to ocean (dark) mode' : 'Switch to daylight (light) mode'}
          aria-label="Toggle theme">
          <span className="theme-toggle-track">
            <span className="theme-toggle-icon sun">
              <svg viewBox="0 0 18 18" width="14" height="14" fill="none">
                <circle cx="9" cy="9" r="3.4" fill="currentColor"/>
                {[0,45,90,135,180,225,270,315].map(a => (
                  <line key={a} x1="9" y1="1.3" x2="9" y2="3.6"
                    stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"
                    transform={`rotate(${a} 9 9)`} />
                ))}
              </svg>
            </span>
            <span className="theme-toggle-icon moon">
              <svg viewBox="0 0 18 18" width="14" height="14" fill="none">
                <path d="M 14.5 11 A 6 6 0 1 1 7 3.5 A 4.5 4.5 0 0 0 14.5 11 Z"
                      fill="currentColor"/>
              </svg>
            </span>
            <span className="theme-toggle-thumb" data-theme={theme} />
          </span>
          <span className="theme-toggle-label">{theme === 'light' ? 'Daylight' : 'Ocean'}</span>
        </button>
      </div>
    </header>
  );
}

function Sidebar() { return null; }

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
          Random Forest model. Selected port: <strong style={{color:'var(--text)'}}>{REGIONS[region].title}</strong>.
        </p>
      </div>
      <div className="hero-meta">
        <div className="hero-meta-row">Forecast as of</div>
        <div className="hero-meta-val">{asOf}</div>
        <div className="hero-meta-row" style={{marginTop:6}}>Model</div>
        <div className="hero-meta-val">{MODEL_LABEL()}</div>
      </div>
    </div>
  );
}

// ====================================================================

function InfoTip({ text }) {
  return (
    <span className="infotip" tabIndex={0} aria-label={text}>
      <svg viewBox="0 0 14 14" width="13" height="13" aria-hidden="true">
        <circle cx="7" cy="7" r="5.6" fill="none" stroke="currentColor" strokeWidth="1.1"/>
        <text x="7" y="10.2" textAnchor="middle" fontFamily="serif" fontSize="9" fill="currentColor" fontWeight="700">?</text>
      </svg>
      <span className="infotip-bubble" role="tooltip">{text}</span>
    </span>
  );
}

function FlagIcon({ country }) {
  if (country === 'Greece') {
    return (
      <svg viewBox="0 0 18 12" width="18" height="12" className="flag-icon" role="img" aria-label="Greece">
        <rect width="18" height="12" fill="#fff" rx="1.6"/>
        <g>
          <rect y="1.33" width="18" height="1.33" fill="#0d5eaf"/>
          <rect y="4"    width="18" height="1.33" fill="#0d5eaf"/>
          <rect y="6.67" width="18" height="1.33" fill="#0d5eaf"/>
          <rect y="9.34" width="18" height="1.33" fill="#0d5eaf"/>
        </g>
        <rect width="6.67" height="6.67" fill="#0d5eaf"/>
        <rect x="2.83" y="0" width="1.0" height="6.67" fill="#fff"/>
        <rect x="0" y="2.83" width="6.67" height="1.0" fill="#fff"/>
      </svg>
    );
  }
  if (country === 'Cyprus') {
    return (
      <svg viewBox="0 0 18 12" width="18" height="12" className="flag-icon" role="img" aria-label="Cyprus">
        <rect width="18" height="12" fill="#fff" rx="1.6" stroke="rgba(10,30,60,0.18)" strokeWidth="0.4"/>
        <path d="M 4.5 4.2 L 7.5 3.4 L 11.5 3.4 L 13.5 4.4 L 13.2 5.6 L 10 5.4 L 8.5 6 L 6 5.6 Z"
              fill="#d57600"/>
        <path d="M 4 9 Q 9 10.5 14 9" stroke="#3f7d3a" strokeWidth="0.45" fill="none" strokeLinecap="round"/>
        <path d="M 5.5 9.6 Q 9 10.8 12.5 9.6" stroke="#3f7d3a" strokeWidth="0.45" fill="none" strokeLinecap="round"/>
      </svg>
    );
  }
  return null;
}

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
          <div className="region-list-title">Monitored ports</div>
          <span style={{fontFamily:'var(--font-mono)', fontSize:10, color:'var(--text-muted)'}}>
            3 / 3 ACTIVE
          </span>
        </div>
        {Object.entries(REGIONS).map(([k, v]) => {
          const sum = regionSummaries[k];
          const level = sum?.level || 'low';
          const trendPct = sum?.chlTrend != null ? sum.chlTrend * 100 : null;
          const trendDir = trendPct == null ? null : (trendPct > 1 ? 'up' : trendPct < -1 ? 'down' : 'flat');
          return (
            <div key={k} className={`region-row ${region===k?'active':''}`}
                 onClick={() => setRegion(k)}>
              <span className="region-row-color" style={{
                background: level==='high' ? 'var(--danger)' : level==='med' ? 'var(--warn)' : 'var(--ok)',
                boxShadow: `0 0 8px ${level==='high' ? 'var(--danger)' : level==='med' ? 'var(--warn)' : 'var(--ok)'}`,
              }} />
              <div className="region-row-content">
                <div className="region-row-name">
                  <FlagIcon country={v.country} />
                  <span>{v.short}</span>
                  {trendDir && (
                    <span className={`region-trend region-trend--${trendDir}`}>
                      {trendDir === 'up' ? '↑' : trendDir === 'down' ? '↓' : '→'}
                      {trendPct != null && ` ${Math.abs(trendPct).toFixed(1)}%`}
                    </span>
                  )}
                </div>
                <div className="region-row-meta">
                  <span>{v.country} · CHL {fmtNum(sum?.chl, 3)} mg/m³</span>
                  {sum?.chlSeries && sum.chlSeries.length > 1 && window.Sparkline && (
                    <span className="region-row-spark">
                      <window.Sparkline
                        values={sum.chlSeries}
                        color={level==='high' ? 'var(--danger)' : level==='med' ? 'var(--warn)' : 'var(--teal)'}
                        height={20}
                        fill={true}
                      />
                    </span>
                  )}
                </div>
              </div>
              <span className={`badge ${level}`}>{sum.tierLabel || mapPinLabel(level)}</span>
            </div>
          );
        })}
        <div style={{padding:'14px 18px', marginTop:'auto', borderTop:'1px solid var(--line)'}}>
          <div style={{fontSize:10.5, color:'var(--text-muted)', letterSpacing:1.2, textTransform:'uppercase', fontWeight:600, marginBottom:6}}>Coverage window</div>
          <div style={{fontFamily:'var(--font-mono)', fontSize:12, color:'var(--text)'}}>
            {cov.start && cov.end ? `${cov.start} → ${cov.end}` : '—'}
          </div>
          <div style={{fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-muted)', marginTop:2}}>
            {cov.forecast_days != null ? `${cov.forecast_days} forecast days` : '—'} · {cov.port_count || cov.basin_count || 3} ports
          </div>
        </div>
      </div>
    </div>
  );
}

// ====================================================================

function KpiCard({ label, value, unit, sub, info, sparkData, sparkColor, badge, badgeLevel }) {
  return (
    <div className="kpi kpi-card">
      <div className="kpi-label">
        <span>{label}</span>
        {info && <InfoTip text={info} />}
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
          value={fmtNum(summary.predicted_chl, 3)}
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
          label="Bloom risk · today"
          value=""
          info="Low / Medium / High / Alert from today's predicted chlorophyll vs the adaptive bloom threshold (P90 of grid forecasts). Model forecast only — not a confirmed bloom in the water."
          badge={bloomRiskTier(summary.risk_score, summary.bloom_flag).label}
          badgeLevel={bloomRiskTier(summary.risk_score, summary.bloom_flag).badgeLevel}
          sub={
            <span>
              {bloomRiskSub(summary.risk_score, summary.predicted_chl, summary.threshold)}
            </span>
          }
        />

        <KpiCard
          label="Adaptive threshold"
          value={fmtNum(summary.threshold, 3)}
          unit="mg/m³"
          sub={<span>P90 of today's grid forecasts</span>}
          sparkData={summary.thresholdSeries}
          sparkColor="var(--warn)"
        />

        <KpiCard
          label="Bloom risk · 7 days"
          value=""
          info="Low / Medium / High from how many of the next 7 forecast days are predicted at or above the bloom threshold."
          badge={outlookRiskLabel(summary.rec7)}
          badgeLevel={outlookRiskLevel(summary.rec7)}
          sub={<span>{fmtNum(summary.rec7, 0)}% · {summary.risk7}/7 days over threshold</span>}
        />

        <KpiCard
          label="Bloom risk · 30 days"
          value=""
          info="Same scale as the 7-day outlook, over the next 30 forecast days."
          badge={outlookRiskLabel(summary.rec30)}
          badgeLevel={outlookRiskLevel(summary.rec30)}
          sub={<span>{fmtNum(summary.rec30, 0)}% · {summary.risk30}/30 days over threshold</span>}
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
        {tab === 'accuracy' && <AccuracyTab region={region} accuracy={accuracy} env={env} />}
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
          ingested per port. We compute lagged values, rolling averages, nutrient ratios and anomalies,
          then feed them to a retrained Random Forest (<code>rf_chl_retrained.pkl</code>). The model emits
          a daily CHL prediction per grid cell, summarised to the port median. A bloom-risk flag fires
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

function isCloseMatch(row) {
  const err = row.err;
  return row.observed_chl != null && !isNaN(row.observed_chl)
    && row.predicted_chl != null && err != null && !isNaN(err)
    && Math.abs(err) < ACCURACY_TOL;
}

function accuracyBadgeLevel(pct) {
  if (pct >= 80) return 'low';
  if (pct >= 50) return 'med';
  return 'high';
}

function windowAccuracy(sorted, n) {
  const slice = sorted.slice(-n);
  if (!slice.length) return null;
  const close = slice.filter(isCloseMatch).length;
  return { pct: Math.round((close / slice.length) * 100), close, total: slice.length };
}

function computeAccuracyStats(rows) {
  const paired = rows.filter(
    d => d.target_date && d.predicted_chl != null && d.observed_chl != null && !isNaN(d.observed_chl)
  );
  if (!paired.length) return null;

  const sorted = [...paired].sort((a, b) => new Date(a.target_date) - new Date(b.target_date));
  const close = sorted.filter(isCloseMatch).length;
  const mae = sorted.reduce((s, d) => s + (d.abs_err || 0), 0) / sorted.length;

  const rolling = [];
  for (let i = 0; i < sorted.length; i++) {
    const win = sorted.slice(Math.max(0, i - 6), i + 1);
    if (win.length < 4) continue;
    rolling.push({
      date: sorted[i].target_date,
      pct: Math.round((win.filter(isCloseMatch).length / win.length) * 100),
    });
  }

  return {
    sorted,
    overall: { pct: Math.round((close / sorted.length) * 100), close, total: sorted.length },
    last7: windowAccuracy(sorted, 7),
    last14: windowAccuracy(sorted, 14),
    mae,
    rolling,
  };
}

function AccuracySkillBlock({ rows, region }) {
  const stats = useMemo(() => computeAccuracyStats(rows), [rows]);
  if (!stats) {
    return (
      <div className="chart" style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
        Not enough paired forecast days to score accuracy yet.
      </div>
    );
  }

  const { overall, last7, last14, mae, rolling } = stats;

  return (
    <div className="accuracy-skill-block">
      <div className="section-head" style={{ marginBottom: 10 }}>
        <h2 className="section-title">Model accuracy · {REGIONS[region].short}</h2>
        <div className="section-sub">
          Predicted vs Copernicus CHL · ±{ACCURACY_TOL} mg/m³
          <InfoTip text={ACCURACY_STATUS_TIP} />
        </div>
      </div>

      <div className="kpi-grid accuracy-kpi-grid">
        <KpiCard
          label="Accuracy · all days"
          badge={`${overall.pct}%`}
          badgeLevel={accuracyBadgeLevel(overall.pct)}
          sub={<span>{overall.close} of {overall.total} days on target</span>}
          info={ACCURACY_STATUS_TIP}
          sparkData={rolling.map(d => d.pct)}
          sparkColor="var(--teal)"
        />
        <KpiCard
          label="Accuracy · 7 days"
          badge={last7 ? `${last7.pct}%` : '—'}
          badgeLevel={last7 ? accuracyBadgeLevel(last7.pct) : 'med'}
          sub={last7 ? <span>{last7.close}/{last7.total} on target</span> : null}
        />
        <KpiCard
          label="Accuracy · 14 days"
          badge={last14 ? `${last14.pct}%` : '—'}
          badgeLevel={last14 ? accuracyBadgeLevel(last14.pct) : 'med'}
          sub={last14 ? <span>{last14.close}/{last14.total} on target</span> : null}
        />
        <KpiCard
          label="Mean absolute error"
          value={fmtNum(mae, 3)}
          unit="mg/m³"
          sub={<span>|predicted − observed| CHL</span>}
          info="Average absolute difference between predicted and observed CHL over all paired days. Lower is better (standard forecast verification metric)."
        />
      </div>

      <div className="chart" style={{ marginTop: 14 }}>
        <div className="chart-header">
          <div className="chart-title">Accuracy trend · 7-day rolling</div>
          <div className="chart-meta">% of days on target (±{ACCURACY_TOL} mg/m³)</div>
        </div>
        <LineChart
          data={rolling}
          x="date"
          ys={[{ key: 'pct', label: 'Accuracy %', color: 'var(--teal)' }]}
          height={200}
          areaFill={true}
          height_unit="%"
        />
      </div>
    </div>
  );
}

function AccuracyTab({ region, accuracy, env }) {
  const merged = useMemo(() => mergeAccuracyWithEnv(accuracy, env), [accuracy, env]);
  const withObs = merged.filter(d => d.observed_chl != null && !isNaN(d.observed_chl) && d.predicted_chl != null);
  const recent = [...merged].sort((a,b) => new Date(b.target_date) - new Date(a.target_date)).slice(0, 12);

  return (
    <div style={{display:'flex', flexDirection:'column', gap:18}}>
      <AccuracySkillBlock rows={merged} region={region} />
      <div style={{display:'grid', gridTemplateColumns:'minmax(0, 1fr) minmax(0, 1fr)', gap:14}}>
        <div className="chart">
          <div className="chart-header">
            <div className="chart-title">Predicted vs Observed · time series</div>
            <div className="chart-meta">
              {REGIONS[region].short} · observed = Copernicus CHL (env-history)
            </div>
          </div>
          <LineChart
            data={merged} x="target_date"
            ys={[
              { key:'predicted_chl', label:'Predicted', color:'var(--depth)' },
              { key:'observed_chl',  label:'Observed',  color:'#e89a2b' },
            ]}
            height={220} areaFill={false}
            height_unit="mg/m³"
          />
        </div>
        <div className="chart">
          <div className="chart-header">
            <div className="chart-title">Predicted vs Observed · scatter</div>
            <div className="chart-meta">n = {withObs.length} paired days</div>
          </div>
          <ScatterChart data={withObs} xKey="predicted_chl" yKey="observed_chl" height={220} />
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'minmax(0, 1fr) minmax(0, 1fr)', gap:14}}>
        <div className="chart" style={{padding:0}}>
          <div className="chart-header" style={{padding:'14px 18px 10px'}}>
            <div className="chart-title">
              Validation metrics · 2023–2025
              <InfoTip text="How well the Random Forest predicted CHL on held-out 2023–2025 data. RMSE / MAE are absolute error in mg/m³ (lower is better); R² is the share of variance explained by the model (1.0 = perfect, 0 = no skill)." />
            </div>
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
                  <td className="num">{(r.n || 0).toLocaleString()}</td>
                  <td className="num">{(r.rmse ?? 0).toFixed(4)}</td>
                  <td className="num">{(r.mae ?? 0).toFixed(4)}</td>
                  <td className="num" style={{color: r.r2 > 0.9 ? 'var(--ok)' : 'var(--text)'}}>
                    {(r.r2 ?? 0).toFixed(4)}
                  </td>
                </tr>
              ))}
              {(() => {
                const t = VALIDATION_TOTAL();
                return (
              <tr className="total">
                <td>{t.region}</td>
                <td className="num">{(t.n || 0).toLocaleString()}</td>
                <td className="num">{(t.rmse ?? 0).toFixed(4)}</td>
                <td className="num">{(t.mae ?? 0).toFixed(4)}</td>
                <td className="num">{(t.r2 ?? 0).toFixed(4)}</td>
              </tr>
                );
              })()}
            </tbody>
          </table>
        </div>

        <div className="chart" style={{padding:0}}>
          <div className="chart-header" style={{padding:'14px 18px 10px'}}>
            <div className="chart-title">
              Bloom detection · precision/recall
              <InfoTip text="Treating bloom-risk as a binary alarm: precision = of the days the model flagged, how many actually crossed the threshold. Recall = of the days that did cross, how many the model caught. F1 balances both; prevalence shows how often blooms occurred in the test window." />
            </div>
            <div className="chart-meta">port thresholds</div>
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
                  <td className="num">{(r.threshold ?? 0).toFixed(2)}</td>
                  <td className="num">{r.prevalence != null ? `${Number(r.prevalence).toFixed(1)}%` : '—'}</td>
                  <td className="num">{(r.precision ?? 0).toFixed(3)}</td>
                  <td className="num">{(r.recall ?? 0).toFixed(3)}</td>
                  <td className="num" style={{color: r.f1 > 0.7 ? 'var(--ok)' : r.f1 > 0 ? 'var(--text)' : 'var(--text-muted)'}}>
                    {(r.f1 ?? 0).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="chart" style={{padding:0}}>
          <div className="chart-header" style={{padding:'14px 18px 10px'}}>
          <div className="chart-title">
            Recent forecasts · {REGIONS[region].short}
            <InfoTip text="Each row is one forecast day. Status = On target if |error| &lt; 0.2 mg/m³ vs Copernicus CHL; Off target if not; Pending if no observation yet." />
          </div>
          <div className="chart-meta">12 most recent days</div>
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th>Date</th>
              <th className="num">Predicted</th>
              <th className="num">Observed</th>
              <th className="num">Error</th>
              <th className="num">|Error|</th>
              <th>
                Status
                <InfoTip text={ACCURACY_STATUS_TIP} />
              </th>
            </tr>
          </thead>
          <tbody>
            {recent.map((r, i) => {
              const hasObs = r.observed_chl != null && !isNaN(r.observed_chl);
              const st = forecastAccuracyStatus(r);
              return (
                <tr key={i}>
                  <td style={{color:'var(--text)'}}>{r.target_date}</td>
                  <td className="num">{fmtNum(r.predicted_chl, 3)}</td>
                  <td className="num">{hasObs ? fmtNum(r.observed_chl, 3) : '—'}</td>
                  <td className="num">{hasObs ? fmtNum(r.err, 3) : '—'}</td>
                  <td className="num">{hasObs ? fmtNum(r.abs_err, 3) : '—'}</td>
                  <td style={{fontFamily:'var(--font-ui)', color: st.color, fontSize: 12}}>
                    {st.label}
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
    const isLight = t.theme === 'light';
    const map = {
      deepblue: isLight
        ? { teal: '#0891b2', depth: '#0c4a6e' }
        : { teal: '#22d3ee', depth: '#7dd3fc' },
      teal:  { teal: '#4dd0c5', depth: '#5c93ff' },
      cyan:  { teal: '#67e8f9', depth: '#3b82f6' },
      kelp:  { teal: '#a3e635', depth: '#0ea5e9' },
      coral: { teal: '#fb7185', depth: '#38bdf8' },
    };
    const c = map[t.accent] || map.deepblue;
    root.style.setProperty('--teal', c.teal);
    root.style.setProperty('--depth', c.depth);
  }, [t.theme, t.accent]);

  const [region, setRegion] = useState('thermaikos');

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

      const tier = bloomRiskTier(last.risk_score, last.flag);
      out[r] = {
        chl: currChl,
        predicted_chl: last.predicted_chl,
        chlTrend: meanChl7 ? (currChl - meanChl7) / meanChl7 : 0,
        temp: currTemp,
        tempTrend: currTemp - meanTemp7,
        threshold: last.threshold,
        risk_score: last.risk_score,
        bloom_flag: last.flag,
        tierLabel: tier.label,
        rec7: (risk7 / 7) * 100,
        rec30: (risk30 / 30) * 100,
        risk7, risk30,
        thresholdSeries: last30.map(d => d.threshold),
        chlSeries: e.slice(-14).map(d => d.chl).filter(v => !isNaN(v)),
        level: tier.badgeLevel,
      };
    });
    return out;
  }, [data]);

  const summary = summaries[region];
  const asOf = forecast[forecast.length - 1]?.date || '—';

  return (
    <div className="app">
      <TopBar
        theme={t.theme}
        onToggleTheme={() => setTweak('theme', t.theme === 'light' ? 'ocean' : 'light')}
      />

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
            <div className="section-sub">3 ports · click any to focus</div>
          </div>
          <MapSection
            region={region}
            setRegion={setRegion}
            regionSummaries={summaries}
            coverage={{
              start: META().coverage_start,
              end: META().coverage_end,
              forecast_days: META().forecast_days,
              port_count: META().port_count ?? META().basin_count,
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
                { value: 'deepblue', label: 'Deep blue' },
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
