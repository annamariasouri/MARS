// MARS Logo — reusable component with 4 variants.
// Pass `variant` to switch. All share the brand palette.
//
// variants:
//   - pulse   : concentric sonar arcs from a center point (satellite sensing the sea)
//   - crest   : letter M built from wave crests
//   - stratum : layered depth bars with a single teal surface line
//   - mark    : refined monogram

function MarsLogo({ variant = 'pulse', size = 40, showBg = true, color = 'var(--teal)', secondary = 'var(--depth)' }) {
  const s = size;

  const bg = showBg ? (
    <>
      <defs>
        <linearGradient id={`mars-bg-${variant}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"  stopColor="#0f2950" />
          <stop offset="100%" stopColor="#061325" />
        </linearGradient>
      </defs>
      <rect x="0.5" y="0.5" width="39" height="39" rx="10" ry="10"
            fill={`url(#mars-bg-${variant})`}
            stroke="rgba(140,180,220,0.15)" strokeWidth="1" />
    </>
  ) : null;

  // viewBox always 40x40, the size attr scales it.

  switch (variant) {

    // ── 1) PULSE — satellite sensing the surface ──────────────────────────
    case 'pulse':
      return (
        <svg width={s} height={s} viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
          {bg}
          {/* Three sonar arcs, expanding outward from sat point */}
          <path d="M 12 22 A 8 8 0 0 1 28 22"
                fill="none" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
          <path d="M 9 22 A 11 11 0 0 1 31 22"
                fill="none" stroke={color} strokeWidth="1.4" strokeLinecap="round" opacity="0.55" />
          <path d="M 6 22 A 14 14 0 0 1 34 22"
                fill="none" stroke={color} strokeWidth="1.4" strokeLinecap="round" opacity="0.28" />
          {/* Surface line + reflective wave */}
          <line x1="6" y1="28" x2="34" y2="28"
                stroke={color} strokeWidth="1.4" strokeLinecap="round" opacity="0.85" />
          <path d="M 8 31.5 Q 12 30 16 31.5 T 24 31.5 T 32 31.5"
                fill="none" stroke={color} strokeWidth="1.1" opacity="0.45" strokeLinecap="round" />
          {/* Satellite / transmitter point */}
          <circle cx="20" cy="22" r="2" fill={secondary} />
          <circle cx="20" cy="22" r="1" fill="#fff" />
        </svg>
      );

    // ── 2) CREST — M from wave crests ─────────────────────────────────────
    case 'crest':
      return (
        <svg width={s} height={s} viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
          {bg}
          {/* The M as two curved peaks meeting at a trough */}
          <path d="M 8 30 L 8 14 Q 8 11 11 14 L 20 23 L 29 14 Q 32 11 32 14 L 32 30"
                fill="none" stroke={color} strokeWidth="2.3"
                strokeLinecap="round" strokeLinejoin="round" />
          {/* Inner wave under the M valley */}
          <path d="M 13 28 Q 16.5 25 20 28 T 27 28"
                fill="none" stroke={color} strokeWidth="1.2" opacity="0.45" strokeLinecap="round" />
          {/* Small depth dot */}
          <circle cx="20" cy="32" r="1.2" fill={secondary} opacity="0.85" />
        </svg>
      );

    // ── 3) STRATUM — depth profile with surface line ──────────────────────
    case 'stratum':
      return (
        <svg width={s} height={s} viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
          {bg}
          {/* Surface (bright teal) */}
          <rect x="8"  y="12" width="24" height="1.6" rx="0.8" fill={color} />
          {/* Subsurface layers */}
          <rect x="10" y="16" width="20" height="1.4" rx="0.7" fill={color} opacity="0.55" />
          <rect x="12" y="19.5" width="16" height="1.3" rx="0.65" fill={color} opacity="0.4" />
          <rect x="9"  y="23"   width="22" height="1.3" rx="0.65" fill={color} opacity="0.3" />
          <rect x="11" y="26.5" width="18" height="1.2" rx="0.6" fill={color} opacity="0.22" />
          <rect x="13" y="30"   width="14" height="1.2" rx="0.6" fill={color} opacity="0.16" />
          {/* Vertical sample / probe line */}
          <line x1="20" y1="9" x2="20" y2="34"
                stroke={secondary} strokeWidth="0.8" opacity="0.7" strokeDasharray="1 1.5" />
          <circle cx="20" cy="9" r="1.6" fill={secondary} />
        </svg>
      );

    // ── 4) MARK — refined monogram with wave undercut ─────────────────────
    case 'mark':
    default:
      return (
        <svg width={s} height={s} viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
          {bg}
          {/* Stylized M */}
          <path d="M 10 28 L 10 13 L 20 23 L 30 13 L 30 28"
                fill="none" stroke={color} strokeWidth="2.4"
                strokeLinecap="round" strokeLinejoin="round" />
          {/* Underline wave */}
          <path d="M 9 32 Q 13 30 17 32 T 25 32 T 31 32"
                fill="none" stroke={color} strokeWidth="1.2" strokeLinecap="round" opacity="0.55" />
          {/* Top dot — satellite/origin */}
          <circle cx="20" cy="10" r="1.5" fill={secondary} />
        </svg>
      );
  }
}

window.MarsLogo = MarsLogo;
