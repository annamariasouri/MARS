// MARS chart components — hand-rolled SVG, no chart library.
// Designed to feel like a marine research console.

const { useState, useMemo, useRef, useEffect } = React;

// ---------- helpers ----------
function niceTicks(min, max, count = 5) {
  const range = max - min;
  if (range === 0) return [min];
  const step0 = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  let step;
  if (norm < 1.5) step = 1 * mag;
  else if (norm < 3) step = 2 * mag;
  else if (norm < 7) step = 5 * mag;
  else step = 10 * mag;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 0.001; v += step) ticks.push(v);
  return ticks;
}

function fmtDate(s) {
  if (!s) return '';
  const d = new Date(s);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function fmtNum(v, p = 2) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(p);
}

// ---------- LineChart ----------
function LineChart({ data, x, ys, height = 220, yLabel, threshold, areaFill = true, height_unit = '' }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const measure = () => setW(el.getBoundingClientRect().width || 600);
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => { ro.disconnect(); window.removeEventListener('resize', measure); };
  }, []);

  const margin = { top: 16, right: 18, bottom: 28, left: 44 };
  const innerW = Math.max(100, w - margin.left - margin.right);
  const innerH = height - margin.top - margin.bottom;

  const valid = data.filter(d => d[x] != null);
  const xVals = valid.map(d => new Date(d[x]).getTime());
  const allY = valid.flatMap(d => ys.map(y => d[y.key])).filter(v => v != null && !isNaN(v));
  if (threshold != null && !isNaN(threshold)) allY.push(threshold);

  if (xVals.length === 0 || allY.length === 0) {
    return <div ref={wrapRef} style={{height, display:'grid', placeItems:'center', color: 'var(--text-muted)', fontSize: 12}}>No data</div>;
  }

  const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  const yPad = (yMax - yMin) * 0.15 || Math.abs(yMax) * 0.1 || 1;
  yMin = Math.max(0, yMin - yPad);
  yMax = yMax + yPad;

  const sx = t => margin.left + ((t - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = v => margin.top + (1 - (v - yMin) / (yMax - yMin || 1)) * innerH;

  const yTicks = niceTicks(yMin, yMax, 4);
  const nXTicks = Math.min(6, valid.length);
  const xTickIdx = Array.from({length: nXTicks}, (_, i) => Math.floor(i * (valid.length - 1) / (nXTicks - 1)));
  const xTicks = xTickIdx.map(i => valid[i][x]);

  return (
    <div ref={wrapRef} style={{ position: 'relative', width: '100%' }}>
      <svg width={w} height={height} style={{ overflow: 'visible', display: 'block' }}>
        {/* Grid */}
        <g className="grid">
          {yTicks.map((v, i) => (
            <line key={i} x1={margin.left} x2={margin.left + innerW} y1={sy(v)} y2={sy(v)} />
          ))}
        </g>
        {/* Axes */}
        <g className="axis">
          {yTicks.map((v, i) => (
            <text key={i} x={margin.left - 8} y={sy(v) + 3} textAnchor="end">{fmtNum(v, v < 10 ? 2 : 0)}</text>
          ))}
          {xTicks.map((t, i) => (
            <text key={i} x={sx(new Date(t).getTime())} y={height - 10} textAnchor="middle">{fmtDate(t)}</text>
          ))}
        </g>

        {/* Threshold line */}
        {threshold != null && !isNaN(threshold) && (
          <g>
            <line
              x1={margin.left} x2={margin.left + innerW}
              y1={sy(threshold)} y2={sy(threshold)}
              stroke="var(--warn)" strokeDasharray="4 4" strokeWidth="1"
              opacity="0.7"
            />
            <text x={margin.left + innerW - 4} y={sy(threshold) - 6}
                  textAnchor="end" fill="var(--warn)"
                  fontSize="10" fontFamily="var(--font-mono)" fontWeight="600">
              threshold {fmtNum(threshold, 2)}
            </text>
          </g>
        )}

        {/* Series */}
        {ys.map((y, yi) => {
          const pts = valid.filter(d => d[y.key] != null && !isNaN(d[y.key]));
          const path = pts.map((d, i) => `${i === 0 ? 'M' : 'L'} ${sx(new Date(d[x]).getTime())} ${sy(d[y.key])}`).join(' ');
          const areaPath = `${path} L ${sx(new Date(pts[pts.length-1][x]).getTime())} ${sy(yMin)} L ${sx(new Date(pts[0][x]).getTime())} ${sy(yMin)} Z`;
          const color = y.color || 'var(--teal)';
          const id = `grad-${y.key}-${yi}`;
          return (
            <g key={y.key}>
              {areaFill && (
                <>
                  <defs>
                    <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={color} stopOpacity="0.3" />
                      <stop offset="100%" stopColor={color} stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d={areaPath} fill={`url(#${id})`} />
                </>
              )}
              <path d={path} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
              {/* End dot */}
              {pts.length > 0 && (() => {
                const last = pts[pts.length - 1];
                return (
                  <g>
                    <circle cx={sx(new Date(last[x]).getTime())} cy={sy(last[y.key])} r="6" fill={color} opacity="0.2" />
                    <circle cx={sx(new Date(last[x]).getTime())} cy={sy(last[y.key])} r="3" fill={color} />
                  </g>
                );
              })()}
            </g>
          );
        })}

        {/* Hover capture */}
        <rect
          x={margin.left} y={margin.top} width={innerW} height={innerH}
          fill="transparent"
          onMouseMove={e => {
            const rect = e.currentTarget.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const t = xMin + (px / innerW) * (xMax - xMin);
            let bestIdx = 0, bestDist = Infinity;
            valid.forEach((d, i) => {
              const dd = Math.abs(new Date(d[x]).getTime() - t);
              if (dd < bestDist) { bestDist = dd; bestIdx = i; }
            });
            setHover({ idx: bestIdx, d: valid[bestIdx] });
          }}
          onMouseLeave={() => setHover(null)}
        />

        {/* Hover line + dots */}
        {hover && (
          <g>
            <line
              x1={sx(new Date(hover.d[x]).getTime())}
              x2={sx(new Date(hover.d[x]).getTime())}
              y1={margin.top} y2={margin.top + innerH}
              stroke="var(--text-dim)" strokeWidth="1" strokeDasharray="2 2" opacity="0.6"
            />
            {ys.map((y, yi) => {
              if (hover.d[y.key] == null) return null;
              return <circle key={yi}
                cx={sx(new Date(hover.d[x]).getTime())}
                cy={sy(hover.d[y.key])}
                r="4" fill="var(--bg)" stroke={y.color} strokeWidth="2" />;
            })}
          </g>
        )}
      </svg>

      {hover && (
        <div className="tt" style={{
          left: ((sx(new Date(hover.d[x]).getTime())) / w * 100) + '%',
          top: margin.top + 'px',
        }}>
          <div className="tt-label">{new Date(hover.d[x]).toLocaleDateString('en-US', { weekday:'short', month:'short', day:'numeric' })}</div>
          {ys.map(y => hover.d[y.key] != null && (
            <div key={y.key} style={{display:'flex', gap:8, alignItems:'center', marginTop:2}}>
              <span style={{display:'inline-block', width:8, height:8, borderRadius:2, background: y.color}} />
              <span className="tt-label" style={{textTransform:'none', fontSize: 11}}>{y.label}</span>
              <span className="tt-val">{fmtNum(hover.d[y.key], 3)} {height_unit}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------- Scatter ----------
function ScatterChart({ data, xKey, yKey, height = 280, color = 'var(--teal)' }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(500);

  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const measure = () => setW(el.getBoundingClientRect().width || 500);
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => { ro.disconnect(); window.removeEventListener('resize', measure); };
  }, []);

  const margin = { top: 14, right: 14, bottom: 36, left: 50 };
  const innerW = Math.max(100, w - margin.left - margin.right);
  const innerH = height - margin.top - margin.bottom;

  const pts = data.filter(d => d[xKey] != null && d[yKey] != null && !isNaN(d[xKey]) && !isNaN(d[yKey]));
  if (pts.length === 0) {
    return <div ref={wrapRef} style={{height, display:'grid', placeItems:'center', color:'var(--text-muted)', fontSize: 12}}>No paired observations</div>;
  }
  const all = [...pts.map(d => d[xKey]), ...pts.map(d => d[yKey])];
  const maxV = Math.max(...all) * 1.05;
  const minV = 0;

  const sx = v => margin.left + ((v - minV) / (maxV - minV)) * innerW;
  const sy = v => margin.top + (1 - (v - minV) / (maxV - minV)) * innerH;
  const ticks = niceTicks(minV, maxV, 5);

  return (
    <div ref={wrapRef} style={{ width: '100%' }}>
      <svg width={w} height={height} style={{ display: 'block' }}>
        <g className="grid">
          {ticks.map((v, i) => (
            <line key={'h'+i} x1={margin.left} x2={margin.left+innerW} y1={sy(v)} y2={sy(v)} />
          ))}
          {ticks.map((v, i) => (
            <line key={'v'+i} y1={margin.top} y2={margin.top+innerH} x1={sx(v)} x2={sx(v)} />
          ))}
        </g>
        <g className="axis">
          {ticks.map((v, i) => (<text key={'yl'+i} x={margin.left-8} y={sy(v)+3} textAnchor="end">{fmtNum(v, 2)}</text>))}
          {ticks.map((v, i) => (<text key={'xl'+i} x={sx(v)} y={height-14} textAnchor="middle">{fmtNum(v, 2)}</text>))}
        </g>
        {/* 1:1 line */}
        <line x1={sx(minV)} y1={sy(minV)} x2={sx(maxV)} y2={sy(maxV)}
          stroke="var(--text-muted)" strokeDasharray="3 4" strokeWidth="1" opacity="0.5" />
        <text x={sx(maxV) - 4} y={sy(maxV) + 14} textAnchor="end" fontSize="10"
              fill="var(--text-muted)" fontFamily="var(--font-mono)">1:1</text>

        {/* points */}
        {pts.map((d, i) => (
          <circle key={i} cx={sx(d[xKey])} cy={sy(d[yKey])} r="4"
            fill={color} fillOpacity="0.35" stroke={color} strokeWidth="1" />
        ))}

        {/* axis labels */}
        <text x={margin.left + innerW/2} y={height-2} textAnchor="middle"
              fontSize="10" fill="var(--text-muted)"
              fontFamily="var(--font-mono)" letterSpacing="1px">
          PREDICTED CHL (mg/m³)
        </text>
        <text x={-(margin.top + innerH/2)} y={12} textAnchor="middle"
              fontSize="10" fill="var(--text-muted)" transform="rotate(-90)"
              fontFamily="var(--font-mono)" letterSpacing="1px">
          OBSERVED CHL (mg/m³)
        </text>
      </svg>
    </div>
  );
}

// ---------- Sparkline ----------
function Sparkline({ values, color = 'var(--teal)', height = 28, fill = true }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(120);
  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const measure = () => setW(el.getBoundingClientRect().width || 120);
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const vs = values.filter(v => v != null && !isNaN(v));
  if (vs.length < 2) return <div ref={wrapRef} style={{height}} />;
  const mn = Math.min(...vs), mx = Math.max(...vs);
  const pad = 2;
  const sx = i => (i / (vs.length-1)) * (w - pad*2) + pad;
  const sy = v => height - pad - ((v - mn) / ((mx - mn) || 1)) * (height - pad*2);
  const path = vs.map((v, i) => `${i===0?'M':'L'} ${sx(i)} ${sy(v)}`).join(' ');
  const area = `${path} L ${sx(vs.length-1)} ${height} L ${sx(0)} ${height} Z`;
  const gid = `spark-${Math.random().toString(36).slice(2,7)}`;
  return (
    <div ref={wrapRef} style={{width:'100%'}}>
      <svg width={w} height={height}>
        {fill && (
          <>
            <defs>
              <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.4" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={area} fill={`url(#${gid})`} />
          </>
        )}
        <path d={path} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
    </div>
  );
}

// ---------- Risk strip ----------
function RiskStrip({ values, threshold }) {
  // values: array of predicted chl values; classify each vs threshold
  return (
    <div className="risk-strip">
      {values.map((v, i) => {
        let cls = 'r0';
        if (v == null) cls = '';
        else if (v >= threshold) cls = 'r3';
        else if (v >= threshold * 0.85) cls = 'r2';
        else cls = 'r1';
        return <div key={i} className={`risk-day ${cls}`} title={fmtNum(v,3)} />;
      })}
    </div>
  );
}

Object.assign(window, { LineChart, ScatterChart, Sparkline, RiskStrip, fmtNum, fmtDate });
